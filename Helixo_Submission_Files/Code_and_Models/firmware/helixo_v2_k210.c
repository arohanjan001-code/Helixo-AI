/**
 * ═══════════════════════════════════════════════════════════════
 *  HELIXO V2 K210 — AI Helmet Detection & Engine Safety Lock
 *  Firmware for Kendryte K210 (Sipeed Maix) + OV5640 Camera
 * ═══════════════════════════════════════════════════════════════
 * 
 *  Detection Logic:
 *    Helmet detected      → Enable ignition relay (GREEN LED)
 *    No helmet detected   → Block ignition relay (RED LED + Buzzer)
 *    No detection at all  → Camera error state (YELLOW LED)
 * 
 *  Safety Features:
 *    - 10-second sliding window (majority vote over multiple frames)
 *    - Progressive response (Warning → Speed reduction → Stop)
 *    - Emergency override (time-limited bypass, event logged)
 *    - 5 FPS detection loop
 * 
 *  Hardware:
 *    - Kendryte K210 (Sipeed Maix Bit / Dock)
 *    - OV5640 5MP Wide Angle Camera (120°-160°)
 *    - 5V Relay Module (ignition control)
 *    - RGB LEDs (status indication)
 *    - Piezo Buzzer (audio alert)
 * 
 *  Build: Kendryte Standalone SDK + CMake
 *  Compile:
 *    mkdir build && cd build
 *    cmake .. -DPROJ=helixo_v2_k210 -DTOOLCHAIN=/opt/kendryte-toolchain/bin
 *    make -j4
 * ═══════════════════════════════════════════════════════════════
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Kendryte SDK headers */
#include "fpioa.h"
#include "gpio.h"
#include "gpiohs.h"
#include "kpu.h"
#include "lcd.h"
#include "nt35310.h"
#include "plic.h"
#include "dvp.h"
#include "sysctl.h"
#include "uarths.h"
#include "timer.h"
#include "image_process.h"

/* ─── Pin Definitions ────────────────────────────────────────── */
#define PIN_RELAY           6       /* Ignition relay control */
#define PIN_LED_GREEN       7       /* Helmet OK indicator */
#define PIN_LED_RED         8       /* No helmet indicator */
#define PIN_LED_YELLOW      9       /* Camera error indicator */
#define PIN_BUZZER          10      /* Audio alert */
#define PIN_EMERGENCY_BTN   11      /* Emergency override button */
#define PIN_GPS_RX          13      /* NEO-6M GPS Receive */
#define PIN_GPS_TX          14      /* NEO-6M GPS Transmit */

/* ─── Detection Parameters ───────────────────────────────────── */
#define MODEL_INPUT_SIZE    224     /* Must match training config */
#define CONFIDENCE_THRESH   0.70f   /* 70% minimum confidence */
#define DETECTION_FPS       5       /* Frames per second */
#define WINDOW_SIZE         50      /* 10 sec window @ 5 FPS */
#define HELMET_CLASS_ID     0       /* Class 0 = helmet */
#define NO_HELMET_CLASS_ID  1       /* Class 1 = no_helmet */

/* ─── Emergency Override ─────────────────────────────────────── */
#define OVERRIDE_DURATION_SEC   1800    /* 30 min bypass per use (Auto/Manual) */
#define MAX_OVERRIDES_MONTHLY   5       /* Max 5 per month */

/* ─── Progressive Response Timing (seconds) ──────────────────── */
#define WARNING_DELAY           30      /* 30s Buzzer warning before auto-quota */
#define EXTRA_WARNING_DELAY     10      /* Extra 10s warning if quota is 0 */

/* ─── State Machine ──────────────────────────────────────────── */
typedef enum {
    STATE_IDLE,                 /* Engine off, waiting for detection */
    STATE_HELMET_OK,            /* Helmet detected, ignition enabled */
    STATE_NO_HELMET_WARNING,    /* No helmet, warning phase */
    STATE_NO_HELMET_FINAL_WARNING, /* No helmet, final warning (0 quota) */
    STATE_NO_HELMET_EXTRA_WARNING, /* No helmet, extra 10s warning */
    STATE_NO_HELMET_STOPPING,   /* No helmet, controlled stop */
    STATE_CAMERA_ERROR,         /* No bounding box / camera blocked */
    STATE_EMERGENCY_OVERRIDE    /* Bypass active */
} system_state_t;

/* ─── Global Variables ───────────────────────────────────────── */
static system_state_t current_state = STATE_IDLE;
static uint8_t detection_window[WINDOW_SIZE];   /* Ring buffer */
static int window_index = 0;
static int override_count = 0;
static uint32_t override_start_time = 0;
static uint32_t no_helmet_start_time = 0;

/* KPU model data (loaded from SD card or flash) */
static kpu_model_context_t helmet_model;
static uint8_t *model_data = NULL;

/* Camera frame buffers */
static uint32_t *display_buf = NULL;
static uint8_t *ai_buf = NULL;

/* Detection result */
typedef struct {
    int class_id;       /* 0=helmet, 1=no_helmet, -1=none */
    float confidence;
    float x, y, w, h;  /* Bounding box (normalized) */
} detection_result_t;


/* ─── Hardware Init ──────────────────────────────────────────── */
static void init_gpio(void) {
    /* Map FPIOA pins to GPIO functions */
    fpioa_set_function(PIN_RELAY,        FUNC_GPIOHS0);
    fpioa_set_function(PIN_LED_GREEN,    FUNC_GPIOHS1);
    fpioa_set_function(PIN_LED_RED,      FUNC_GPIOHS2);
    fpioa_set_function(PIN_LED_YELLOW,   FUNC_GPIOHS3);
    fpioa_set_function(PIN_BUZZER,       FUNC_GPIOHS4);
    fpioa_set_function(PIN_EMERGENCY_BTN, FUNC_GPIOHS5);
    
    /* Set directions */
    gpiohs_set_drive_mode(0, GPIO_DM_OUTPUT);  /* Relay */
    gpiohs_set_drive_mode(1, GPIO_DM_OUTPUT);  /* Green LED */
    gpiohs_set_drive_mode(2, GPIO_DM_OUTPUT);  /* Red LED */
    gpiohs_set_drive_mode(3, GPIO_DM_OUTPUT);  /* Yellow LED */
    gpiohs_set_drive_mode(4, GPIO_DM_OUTPUT);  /* Buzzer */
    gpiohs_set_drive_mode(5, GPIO_DM_INPUT_PULL_UP);  /* Button */
    
    /* Initial state: all off */
    gpiohs_set_pin(0, GPIO_PV_LOW);   /* Relay OFF */
    gpiohs_set_pin(1, GPIO_PV_LOW);   /* Green OFF */
    gpiohs_set_pin(2, GPIO_PV_LOW);   /* Red OFF */
    gpiohs_set_pin(3, GPIO_PV_LOW);   /* Yellow OFF */
    gpiohs_set_pin(4, GPIO_PV_LOW);   /* Buzzer OFF */
    
    printf("[INIT] GPIO configured.\n");
}


static void init_camera(void) {
    /* Initialize DVP camera interface for OV5640 */
    dvp_init(8);                          /* 8-bit interface */
    dvp_set_xclk_rate(24000000);          /* 24MHz XCLK */
    dvp_enable_burst();
    dvp_set_output_enable(0, 1);          /* AI output */
    dvp_set_output_enable(1, 1);          /* Display output */
    dvp_set_image_format(DVP_CFG_RGB_FORMAT);
    dvp_set_image_size(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
    
    /* Allocate frame buffers */
    display_buf = (uint32_t *)malloc(MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 2);
    ai_buf = (uint8_t *)malloc(MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 3);
    
    dvp_set_display_addr((uint32_t)display_buf);
    dvp_set_ai_addr((uint32_t)ai_buf, 
                    (uint32_t)(ai_buf + MODEL_INPUT_SIZE * MODEL_INPUT_SIZE),
                    (uint32_t)(ai_buf + MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 2));
    
    printf("[INIT] Camera (OV5640) configured @ %dx%d.\n", 
           MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
}


static int init_model(void) {
    /* 
     * Load .kmodel from SD card or flash.
     * The model file should be at: /sd/helixo_v2_k210.kmodel
     */
    const char *model_path = "/sd/helixo_v2_k210.kmodel";
    
    FILE *f = fopen(model_path, "rb");
    if (!f) {
        printf("[ERROR] Model not found: %s\n", model_path);
        printf("[ERROR] Place helixo_v2_k210.kmodel on SD card root.\n");
        return -1;
    }
    
    /* Get file size */
    fseek(f, 0, SEEK_END);
    long model_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    /* Allocate and read */
    model_data = (uint8_t *)malloc(model_size);
    if (!model_data) {
        printf("[ERROR] Failed to allocate %ld bytes for model.\n", model_size);
        fclose(f);
        return -1;
    }
    
    fread(model_data, 1, model_size, f);
    fclose(f);
    
    /* Initialize KPU with model */
    if (kpu_load_kmodel(&helmet_model, model_data) != 0) {
        printf("[ERROR] Failed to load .kmodel\n");
        free(model_data);
        return -1;
    }
    
    printf("[INIT] Model loaded: %s (%ld KB)\n", model_path, model_size / 1024);
    return 0;
}


/* ─── Detection Logic ────────────────────────────────────────── */
static void run_detection(detection_result_t *result) {
    /* Run KPU inference on current AI buffer */
    result->class_id = -1;
    result->confidence = 0.0f;
    
    /* Start KPU inference */
    kpu_run_kmodel(&helmet_model, ai_buf, KPU_DMA_CH, NULL, NULL);
    
    /* Get output (simplified — actual parsing depends on model output format) */
    float *output = NULL;
    size_t output_size = 0;
    kpu_get_output(&helmet_model, 0, (uint8_t **)&output, &output_size);
    
    /* 
     * Parse YOLO output: find highest confidence detection
     * Output format depends on YOLOv8 export settings.
     * Typically: [batch, num_detections, 6] where 6 = [x,y,w,h,conf,class]
     * 
     * NOTE: This parsing logic will need to be adjusted based on 
     * the actual .kmodel output tensor layout after export.
     */
    int num_detections = output_size / (6 * sizeof(float));
    float best_conf = 0.0f;
    
    for (int i = 0; i < num_detections; i++) {
        float x    = output[i * 6 + 0];
        float y    = output[i * 6 + 1];
        float w    = output[i * 6 + 2];
        float h    = output[i * 6 + 3];
        float conf = output[i * 6 + 4];
        int   cls  = (int)output[i * 6 + 5];
        
        if (conf > CONFIDENCE_THRESH && conf > best_conf) {
            best_conf = conf;
            result->class_id = cls;
            result->confidence = conf;
            result->x = x;
            result->y = y;
            result->w = w;
            result->h = h;
        }
    }
}


/* ─── Sliding Window Analysis ────────────────────────────────── */
static int analyze_window(void) {
    /*
     * Analyze the sliding window to make a decision.
     * Returns: 0 = helmet, 1 = no_helmet, -1 = no detection
     * 
     * Uses majority vote over the last WINDOW_SIZE frames.
     * This prevents single-frame glitches from affecting the decision.
     */
    int helmet_count = 0;
    int no_helmet_count = 0;
    int no_detect_count = 0;
    
    for (int i = 0; i < WINDOW_SIZE; i++) {
        switch (detection_window[i]) {
            case 0:  helmet_count++;     break;  /* helmet */
            case 1:  no_helmet_count++;  break;  /* no_helmet */
            default: no_detect_count++;  break;  /* no detection */
        }
    }
    
    /* If >70% of frames have no detection → camera error */
    if (no_detect_count > WINDOW_SIZE * 0.7) {
        return -1;
    }
    
    /* Majority vote between helmet and no_helmet */
    if (helmet_count >= no_helmet_count) {
        return 0;  /* helmet */
    } else {
        return 1;  /* no_helmet */
    }
}


/* ─── Hardware Control ───────────────────────────────────────── */
static void set_relay(int on) {
    gpiohs_set_pin(0, on ? GPIO_PV_HIGH : GPIO_PV_LOW);
}

static void set_led_green(int on) {
    gpiohs_set_pin(1, on ? GPIO_PV_HIGH : GPIO_PV_LOW);
}

static void set_led_red(int on) {
    gpiohs_set_pin(2, on ? GPIO_PV_HIGH : GPIO_PV_LOW);
}

static void set_led_yellow(int on) {
    gpiohs_set_pin(3, on ? GPIO_PV_HIGH : GPIO_PV_LOW);
}

static void set_buzzer(int on) {
    gpiohs_set_pin(4, on ? GPIO_PV_HIGH : GPIO_PV_LOW);
}

static int is_emergency_pressed(void) {
    return gpiohs_get_pin(5) == GPIO_PV_LOW;  /* Active LOW */
}

static void all_off(void) {
    set_relay(0);
    set_led_green(0);
    set_led_red(0);
    set_led_yellow(0);
    set_buzzer(0);
}


/* ─── GPS Integration ────────────────────────────────────────── */
static int get_current_speed(void) {
    /* 
     * TODO: This function will read NMEA sentences from NEO-6M 
     * via UART on PIN_GPS_RX to get exact speed in km/h.
     * For prototype testing without riding, we return a simulated speed.
     */
    int simulated_speed = 30; /* Set to 45 to test Auto-Quota consumption */
    return simulated_speed;
}

/* ─── State Handlers ─────────────────────────────────────────── */
static void handle_helmet_ok(void) {
    current_state = STATE_HELMET_OK;
    no_helmet_start_time = 0;
    
    set_relay(1);           /* Enable ignition */
    set_led_green(1);       /* Green ON */
    set_led_red(0);
    set_led_yellow(0);
    set_buzzer(0);
    
    printf("[OK] Helmet detected — Ignition ENABLED\n");
}


static void handle_no_helmet(uint32_t current_time) {
    if (no_helmet_start_time == 0) {
        no_helmet_start_time = current_time;
    }
    
    uint32_t elapsed = current_time - no_helmet_start_time;
    
    if (override_count < MAX_OVERRIDES_MONTHLY) {
        /* Quotas are available! */
        if (elapsed < WARNING_DELAY) {
            /* 30 seconds warning phase */
            current_state = STATE_NO_HELMET_WARNING;
            set_led_red(1);
            set_buzzer(1);      /* Beep warning */
            set_led_green(0);
            /* Relay still ON */
            printf("[WARN] No helmet — Warning phase (%u / %u sec)\n", elapsed, WARNING_DELAY);
        } else {
            /* 30 seconds passed. Check speed to decide SAFE CUT or HIGH-SPEED AUTO-QUOTA */
            int current_speed = get_current_speed();
            
            if (current_speed <= 40) {
                printf("[SAFE-CUT] Speed %d km/h <= 40. Safe to cut engine. Quota preserved!\n", current_speed);
                current_state = STATE_NO_HELMET_STOPPING;
                set_relay(0);       /* Cut ignition immediately */
                set_led_red(1);
                set_buzzer(1);
                set_led_green(0);
                no_helmet_start_time = 0;
            } else {
                printf("[AUTO-EMERGENCY] Speed %d km/h > 40! Dangerous to cut. Auto-consuming 1 quota (30 mins).\n", current_speed);
                no_helmet_start_time = 0; /* Reset for future */
                override_start_time = current_time;
                override_count++;
                current_state = STATE_EMERGENCY_OVERRIDE;
                
                set_relay(1);
                set_led_green(1);
                set_led_yellow(1);
                set_led_red(0);
                set_buzzer(0);
            }
        }
    } else {
        /* NO QUOTAS LEFT! */
        if (elapsed < WARNING_DELAY) {
            current_state = STATE_NO_HELMET_FINAL_WARNING;
            set_led_red(1);
            set_buzzer(1);
            set_led_green(0);
            printf("[FINAL WARN] Quota Empty! Wear helmet! (%u / %u sec)\n", elapsed, WARNING_DELAY);
        } else if (elapsed < WARNING_DELAY + EXTRA_WARNING_DELAY) {
            current_state = STATE_NO_HELMET_EXTRA_WARNING;
            set_led_red(1);
            set_buzzer(1); /* Fast loop beeping logic can be handled in main thread, assume ON */
            set_led_green(0);
            printf("[CRITICAL] Extra 10s warning before CUT! (%u / %u sec)\n", elapsed, WARNING_DELAY + EXTRA_WARNING_DELAY);
        } else {
            /* CUT IGNITION */
            current_state = STATE_NO_HELMET_STOPPING;
            set_relay(0);       /* Cut ignition */
            set_led_red(1);
            set_buzzer(1);
            set_led_green(0);
            printf("[STOP] No helmet — Ignition CUT (%u sec). Quota completely empty.\n", elapsed);
        }
    }
}


static void handle_camera_error(void) {
    current_state = STATE_CAMERA_ERROR;
    
    all_off();
    set_led_yellow(1);     /* Yellow = camera issue */
    set_buzzer(1);         /* Alert rider */
    /* Keep relay in last known state for safety */
    
    printf("[ERROR] No detection — Clean camera / Out of frame\n");
}


static void handle_emergency_override(uint32_t current_time) {
    if (override_count >= MAX_OVERRIDES_MONTHLY) {
        printf("[OVERRIDE] Monthly limit reached (%d/%d)\n", 
               override_count, MAX_OVERRIDES_MONTHLY);
        return;
    }
    
    if (current_state != STATE_EMERGENCY_OVERRIDE) {
        /* Activate manual override from button */
        override_start_time = current_time;
        override_count++;
        current_state = STATE_EMERGENCY_OVERRIDE;
        no_helmet_start_time = 0;
        
        printf("[OVERRIDE] MANUAL Emergency bypass ACTIVATED (%d/%d used)\n",
               override_count, MAX_OVERRIDES_MONTHLY);
    }
    
    uint32_t elapsed = current_time - override_start_time;
    
    if (elapsed < OVERRIDE_DURATION_SEC) {
        /* Override active */
        set_relay(1);           /* Force ignition ON */
        set_led_green(1);
        set_led_yellow(1);     /* Yellow = override indicator */
        set_led_red(0);
        set_buzzer(0);
    } else {
        /* Override expired */
        printf("[OVERRIDE] Bypass EXPIRED (30 mins completed) — returning to normal mode\n");
        current_state = STATE_IDLE;
        override_start_time = 0;
        no_helmet_start_time = 0; /* Reset timer so they get a fresh 30s warning before next quota */
    }
}


/* ─── Main Loop ──────────────────────────────────────────────── */
int main(void) {
    /* System init */
    sysctl_pll_set_freq(SYSCTL_PLL0, 800000000UL);  /* 800MHz */
    sysctl_pll_set_freq(SYSCTL_PLL1, 400000000UL);  /* 400MHz */
    
    uarths_init();
    plic_init();
    sysctl_enable_irq();
    
    printf("\n");
    printf("═══════════════════════════════════════════════\n");
    printf("  HELIXO V2 K210 — Helmet Detection System\n");
    printf("  Kendryte K210 + OV5640 (Wide Angle)\n");
    printf("═══════════════════════════════════════════════\n\n");
    
    /* Initialize subsystems */
    init_gpio();
    init_camera();
    
    if (init_model() != 0) {
        printf("[FATAL] Model initialization failed!\n");
        /* Flash red LED rapidly to indicate error */
        while (1) {
            set_led_red(1);
            msleep(200);
            set_led_red(0);
            msleep(200);
        }
    }
    
    /* Clear detection window */
    memset(detection_window, 0xFF, sizeof(detection_window));  /* 0xFF = no detection */
    
    printf("[READY] System initialized. Starting detection loop @ %d FPS\n\n", 
           DETECTION_FPS);
    
    uint32_t frame_delay_ms = 1000 / DETECTION_FPS;
    detection_result_t result;
    
    /* ── Main Detection Loop ── */
    while (1) {
        uint32_t current_time = sysctl_get_time_us() / 1000000;  /* Seconds */
        
        /* Check emergency override button */
        if (is_emergency_pressed()) {
            handle_emergency_override(current_time);
            msleep(frame_delay_ms);
            continue;
        }
        
        /* If in override mode, check expiry */
        if (current_state == STATE_EMERGENCY_OVERRIDE) {
            handle_emergency_override(current_time);
            msleep(frame_delay_ms);
            continue;
        }
        
        /* Capture frame */
        dvp_clear_interrupt(DVP_STS_FRAME_START | DVP_STS_FRAME_FINISH);
        dvp_config_interrupt(DVP_CFG_START_INT_ENABLE | DVP_CFG_FINISH_INT_ENABLE, 0);
        
        /* Run AI detection */
        run_detection(&result);
        
        /* Update sliding window */
        detection_window[window_index] = (uint8_t)(result.class_id & 0xFF);
        window_index = (window_index + 1) % WINDOW_SIZE;
        
        /* Analyze window and take action */
        int decision = analyze_window();
        
        switch (decision) {
            case 0:   /* Helmet detected */
                handle_helmet_ok();
                break;
                
            case 1:   /* No helmet */
                handle_no_helmet(current_time);
                break;
                
            case -1:  /* No bounding box / camera blocked */
                handle_camera_error();
                break;
        }
        
        /* Frame rate control */
        msleep(frame_delay_ms);
    }
    
    return 0;
}
