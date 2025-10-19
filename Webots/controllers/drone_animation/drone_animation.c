#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <webots/robot.h>
#include <webots/supervisor.h>
#include <webots/motor.h>
#include <webots/keyboard.h>

#include "sensor_detection.h"

#define START_X -5.0
#define END_X 1000.0
#define BASE_ALTITUDE 7.5
#define DRONE_SPEED 10.25
#define BALL_DROP_INTERVAL 100.0
#define MAX_BALLS 15

#define DRONE2_START_Y -200.0
#define DRONE2_END_Y 200.0
#define DRONE2_X 500.0  
#define DRONE2_ALTITUDE 7.5
#define DRONE2_SPEED 6.25

#define SENSOR_RANGE 200.0
#define MAX_SENSORS 15

double multiOctaveNoise(double x, double base_freq, double base_amp, int octaves) {
    double total = 0.0;
    double amplitude = base_amp;
    double frequency = base_freq;

    for (int i = 0; i < octaves; i++) {
        total += amplitude * sin(x * frequency);
        total += amplitude * 0.5 * sin(x * frequency * 2.3);
        total += amplitude * 0.3 * cos(x * frequency * 1.7);
        amplitude *= 0.5;
        frequency *= 2.0;
    }

    return total;
}

int main(int argc, char **argv) {
    wb_robot_init();
    int timestep = (int)wb_robot_get_basic_time_step();

    double distance = END_X - START_X;
    double flight_duration = distance / DRONE_SPEED;

    printf("=== Drone Animation Controller ===\n");
    printf("Speed: %.2f m/s | Distance: %.1fm | Duration: %.1fs\n\n",
           DRONE_SPEED, distance, flight_duration);

    WbNodeRef drone_node = wb_supervisor_node_get_self();
    if (!drone_node) {
        printf("ERROR: Could not get drone node!\n");
        return 1;
    }

    WbFieldRef trans_field = wb_supervisor_node_get_field(drone_node, "translation");
    WbFieldRef rotation_field = wb_supervisor_node_get_field(drone_node, "rotation");
    
    WbDeviceTag front_left_motor = wb_robot_get_device("front left propeller");
    WbDeviceTag front_right_motor = wb_robot_get_device("front right propeller");
    WbDeviceTag rear_left_motor = wb_robot_get_device("rear left propeller");
    WbDeviceTag rear_right_motor = wb_robot_get_device("rear right propeller");

    if (front_left_motor) {
        wb_motor_set_position(front_left_motor, INFINITY);
        wb_motor_set_velocity(front_left_motor, 150.0);
    }
    if (front_right_motor) {
        wb_motor_set_position(front_right_motor, INFINITY);
        wb_motor_set_velocity(front_right_motor, -150.0);
    }
    if (rear_left_motor) {
        wb_motor_set_position(rear_left_motor, INFINITY);
        wb_motor_set_velocity(rear_left_motor, 150.0);
    }
    if (rear_right_motor) {
        wb_motor_set_position(rear_right_motor, INFINITY);
        wb_motor_set_velocity(rear_right_motor, -150.0);
    }

    const double initial_pos[3] = {START_X, 0.0, BASE_ALTITUDE};
    wb_supervisor_field_set_sf_vec3f(trans_field, initial_pos);
    for (int i = 0; i < 10; i++) {
        wb_robot_step(timestep);
    }

    WbNodeRef vp_main = wb_supervisor_node_get_from_def("MAIN_VP");
    wb_keyboard_enable(timestep);

    int current_view = 0;

    
    int ball_count = 0;
    double next_drop_distance = BALL_DROP_INTERVAL;
    WbNodeRef last_ball = NULL;

    
    Sensor sensors[MAX_SENSORS];
    for (int i = 0; i < MAX_SENSORS; i++) {
        sensors[i].node = NULL;
        sensors[i].active = false;
        sensors[i].signal_strength = 0.0;
    }

    WbNodeRef drone2 = wb_supervisor_node_get_from_def("DRONE2");
    WbFieldRef drone2_trans_field = NULL;
    WbFieldRef drone2_rotation_field = NULL;

    if (drone2) {
        drone2_trans_field = wb_supervisor_node_get_field(drone2, "translation");
        drone2_rotation_field = wb_supervisor_node_get_field(drone2, "rotation");

        const double drone2_initial_pos[3] = {DRONE2_X, DRONE2_START_Y, DRONE2_ALTITUDE};
        wb_supervisor_field_set_sf_vec3f(drone2_trans_field, drone2_initial_pos);
    }

    bool drone1_finished = false;
    double drone2_start_time = 0.0;
    WbNodeRef detection_zone = NULL;
    bool drone2_finished = false;

    printf("Camera Controls:\n");
    printf("  1 - Free camera\n  2 - Drone FPV\n  3 - Follow\n  4 - Front\n  5 - Case Follow\n\n");
    printf("Starting flight...\n");

    double start_time = wb_robot_get_time();
    double last_print = 0.0;
    double free_cam_pos[3] = {18, -15, 8};
    double free_cam_orient[4] = {-0.3, 0.2, 0.93, 2.0};
    bool free_cam_initialized = false;
    (void)free_cam_pos;
    (void)free_cam_orient;

    while (wb_robot_step(timestep) != -1) {
        double elapsed = wb_robot_get_time() - start_time;
        double distance_traveled = DRONE_SPEED * elapsed;
        double x = START_X + distance_traveled;

        if (x > END_X) {
            x = END_X;
        }

        double progress = (x - START_X) / distance;

        int key = wb_keyboard_get_key();
        if (key == '1' && current_view != 0) {
            if (vp_main && !free_cam_initialized) {
                WbFieldRef main_trans = wb_supervisor_node_get_field(vp_main, "position");
                WbFieldRef main_rot = wb_supervisor_node_get_field(vp_main, "orientation");
                const double *current_pos = wb_supervisor_field_get_sf_vec3f(main_trans);
                const double *current_rot = wb_supervisor_field_get_sf_rotation(main_rot);
                free_cam_pos[0] = current_pos[0];
                free_cam_pos[1] = current_pos[1];
                free_cam_pos[2] = current_pos[2];
                free_cam_orient[0] = current_rot[0];
                free_cam_orient[1] = current_rot[1];
                free_cam_orient[2] = current_rot[2];
                free_cam_orient[3] = current_rot[3];
                free_cam_initialized = true;
            }
            current_view = 0;
            printf("Switched to: Free camera\n");
        } else if (key == '2' && current_view != 1) {
            current_view = 1;
            printf("Switched to: Drone FPV\n");
        } else if (key == '3' && current_view != 2) {
            current_view = 2;
            printf("Switched to: Follow camera\n");
        } else if (key == '4' && current_view != 3) {
            current_view = 3;
            printf("Switched to: Front camera\n");
        } else if (key == '5' && current_view != 4) {
            current_view = 4;
            printf("Switched to: Ball Follow camera\n");
        }

        double altitude_wave = 0.6 * sin(progress * M_PI * 2.0);
        double y_drift = multiOctaveNoise(elapsed, 0.4, 0.3, 3);
        double z_noise = multiOctaveNoise(elapsed + 100.0, 0.6, 0.15, 3);
        double y = y_drift;
        double z = BASE_ALTITUDE + altitude_wave + z_noise;

        const double new_pos[3] = {x, y, z};
        wb_supervisor_field_set_sf_vec3f(trans_field, new_pos);

        double roll = -y_drift * 0.2;
        double pitch = -0.05;
        double yaw = multiOctaveNoise(elapsed + 200.0, 0.25, 0.1, 2);

        const double new_rotation[4] = {
            roll * 0.3,
            1.0 - fabs(roll) * 0.2,
            0.1,
            pitch + roll * 0.5 + yaw * 0.3
        };
        wb_supervisor_field_set_sf_rotation(rotation_field, new_rotation);

        if (distance_traveled >= next_drop_distance && ball_count < MAX_BALLS) {
            char case_def[64];
            snprintf(case_def, sizeof(case_def), "CASE_%d", ball_count + 1);

            char case_string[256];
            snprintf(case_string, sizeof(case_string),
                "DEF %s CaseDeploy {\n"
                "  translation %.2f %.2f %.2f\n"
                "  mass 2.0\n"
                "  hitboxRadius 0.3\n"
                "}\n",
                case_def, x, y, z);

            WbNodeRef root = wb_supervisor_node_get_root();
            WbFieldRef children = wb_supervisor_node_get_field(root, "children");
            wb_supervisor_field_import_mf_node_from_string(children, -1, case_string);

            last_ball = wb_supervisor_node_get_from_def(case_def);

            
            sensors[ball_count].node = last_ball;
            sensors[ball_count].position[0] = x;
            sensors[ball_count].position[1] = y;
            sensors[ball_count].position[2] = 0.0;
            sensors[ball_count].active = true;

            ball_count++;

            printf("CaseDeploy/Sensor dropped at X=%.1f! (Sensor %d/%d)\n", x, ball_count, MAX_BALLS);
            next_drop_distance += BALL_DROP_INTERVAL;
        }

        
        if (x >= END_X && !drone1_finished) {
            drone1_finished = true;
            drone2_start_time = wb_robot_get_time();
            printf("\n=== Drone 1 completed! Starting Drone 2 ===\n");
            printf("Deployed %d sensors with 200m detection range\n\n", ball_count);
        }

        
        if (drone1_finished && drone2 && !drone2_finished) {
            double drone2_elapsed = wb_robot_get_time() - drone2_start_time;
            double drone2_distance = DRONE2_SPEED * drone2_elapsed;
            double drone2_y = DRONE2_START_Y + drone2_distance;

            if (drone2_y <= DRONE2_END_Y) {
                
                double y_noise = multiOctaveNoise(drone2_elapsed, 0.4, 0.2, 2);
                double z_noise = multiOctaveNoise(drone2_elapsed + 300.0, 0.6, 0.15, 2);

                double drone2_x = DRONE2_X;
                double drone2_z = DRONE2_ALTITUDE + z_noise;

                const double drone2_pos[3] = {drone2_x, drone2_y + y_noise, drone2_z};
                wb_supervisor_field_set_sf_vec3f(drone2_trans_field, drone2_pos);

                double roll2 = -y_noise * 0.15;
                const double drone2_rotation[4] = {
                    roll2 * 0.3,
                    1.0 - fabs(roll2) * 0.2,
                    0.1,
                    1.5708
                };
                wb_supervisor_field_set_sf_rotation(drone2_rotation_field, drone2_rotation);

                for (int i = 0; i < ball_count; i++) {
                    if (sensors[i].node) {
                        WbFieldRef sensor_trans = wb_supervisor_node_get_field(sensors[i].node, "translation");
                        const double *actual_pos = wb_supervisor_field_get_sf_vec3f(sensor_trans);
                        sensors[i].position[0] = actual_pos[0];
                        sensors[i].position[1] = actual_pos[1];
                        sensors[i].position[2] = actual_pos[2];
                    }
                }

                
                int detecting_sensors[2] = {-1, -1};
                int detection_count = 0;

                for (int i = 0; i < ball_count && detection_count < 2; i++) {
                    if (sensors[i].active) {
                        double dist = distance_3d(sensors[i].position, drone2_pos);
                        if (dist <= SENSOR_RANGE) {
                            sensors[i].signal_strength = calculate_signal_strength(dist);
                            detecting_sensors[detection_count] = i;
                            detection_count++;
                        } else {
                            sensors[i].signal_strength = 0.0;
                        }
                    }
                }

                
                if (detection_count == 2) {
                    double estimated_pos[3];
                    double zone_radius;
                    estimate_detection_zone(&sensors[detecting_sensors[0]],
                                          &sensors[detecting_sensors[1]],
                                          estimated_pos, &zone_radius);

                    if (detection_zone == NULL) {
                        char zone_string[1024];
                        snprintf(zone_string, sizeof(zone_string),
                            "DEF DETECT_ZONE Transform {\n"
                            "  translation %.2f %.2f %.2f\n"
                            "  children [\n"
                            "    DEF ZONE_SHAPE Shape {\n"
                            "      appearance PBRAppearance {\n"
                            "        baseColor 1 1 0\n"
                            "        transparency 0.7\n"
                            "        metalness 0\n"
                            "      }\n"
                            "      geometry DEF ZONE_SPHERE Sphere {\n"
                            "        radius %.2f\n"
                            "      }\n"
                            "    }\n"
                            "  ]\n"
                            "}\n",
                            estimated_pos[0], estimated_pos[1], estimated_pos[2], zone_radius);

                        WbNodeRef root = wb_supervisor_node_get_root();
                        WbFieldRef children = wb_supervisor_node_get_field(root, "children");
                        wb_supervisor_field_import_mf_node_from_string(children, -1, zone_string);
                        detection_zone = wb_supervisor_node_get_from_def("DETECT_ZONE");
                    } else {
                        WbFieldRef zone_trans = wb_supervisor_node_get_field(detection_zone, "translation");
                        wb_supervisor_field_set_sf_vec3f(zone_trans, estimated_pos);

                        WbNodeRef zone_sphere = wb_supervisor_node_get_from_def("ZONE_SPHERE");
                        if (zone_sphere) {
                            WbFieldRef radius_field = wb_supervisor_node_get_field(zone_sphere, "radius");
                            wb_supervisor_field_set_sf_float(radius_field, zone_radius);
                        }
                    }

                    printf("DETECTION: Sensors %d,%d | Drone2: (%.1f,%.1f) | Zone: (%.1f,%.1f) r=%.1fm\n",
                           detecting_sensors[0]+1, detecting_sensors[1]+1,
                           drone2_pos[0], drone2_pos[1],
                           estimated_pos[0], estimated_pos[1], zone_radius);
                }
            } else {
                drone2_finished = true;
                printf("\n=== Drone 2 completed perpendicular flight! ===\n");
                printf("Final Drone 2 position: (%.2f, %.2f, %.2f)\n", DRONE2_X, drone2_y, DRONE2_ALTITUDE);
            }
        }

        if (current_view == 0) {

        } else if (current_view == 1 && vp_main) {
            WbFieldRef main_trans = wb_supervisor_node_get_field(vp_main, "position");
            WbFieldRef main_rot = wb_supervisor_node_get_field(vp_main, "orientation");

            const double fpv_pos[3] = {x + 0.1, y, z + 0.1};
            wb_supervisor_field_set_sf_vec3f(main_trans, fpv_pos);

            const double fpv_orient[4] = {0.0, 0.0, 1.0, 0.0};
            wb_supervisor_field_set_sf_rotation(main_rot, fpv_orient);

        } else if (current_view == 2 && vp_main) {
            WbFieldRef main_trans = wb_supervisor_node_get_field(vp_main, "position");
            WbFieldRef main_rot = wb_supervisor_node_get_field(vp_main, "orientation");

            double cam_shake_x = multiOctaveNoise(elapsed + 50.0, 2.5, 0.04, 2);
            double cam_shake_y = multiOctaveNoise(elapsed + 150.0, 2.8, 0.04, 2);
            double cam_shake_z = multiOctaveNoise(elapsed + 250.0, 3.0, 0.03, 2);

            const double follow_pos[3] = {
                x - 2.0 + cam_shake_x,
                y + cam_shake_y,
                z + 0.2 + cam_shake_z
            };
            wb_supervisor_field_set_sf_vec3f(main_trans, follow_pos);

            double dx_look = x - follow_pos[0];
            double dy_look = y - follow_pos[1];
            double dz_look = z - follow_pos[2];
            double dist = sqrt(dx_look*dx_look + dy_look*dy_look + dz_look*dz_look);

            dx_look /= dist;
            dy_look /= dist;
            dz_look /= dist;

            double pitch_angle = atan2(-dz_look, sqrt(dx_look*dx_look + dy_look*dy_look));
            double pitch_shake = multiOctaveNoise(elapsed + 350.0, 3.2, 0.01, 2);

            const double follow_orient[4] = {
                0.0, 1.0, 0.0, pitch_angle + 0 + pitch_shake
            };
            wb_supervisor_field_set_sf_rotation(main_rot, follow_orient);

        } else if (current_view == 3 && vp_main) {
            WbFieldRef main_trans = wb_supervisor_node_get_field(vp_main, "position");
            WbFieldRef main_rot = wb_supervisor_node_get_field(vp_main, "orientation");

            double cam_shake_x = multiOctaveNoise(elapsed + 500.0, 2.6, 0.04, 2);
            double cam_shake_y = multiOctaveNoise(elapsed + 600.0, 2.9, 0.04, 2);
            double cam_shake_z = multiOctaveNoise(elapsed + 700.0, 3.1, 0.03, 2);

            const double front_pos[3] = {
                x + 2.0 + cam_shake_x,
                y + cam_shake_y,
                z + 0.2 + cam_shake_z
            };
            wb_supervisor_field_set_sf_vec3f(main_trans, front_pos);

            double dx_look = x - front_pos[0];
            double dy_look = y - front_pos[1];
            double dz_look = z - front_pos[2];
            double dist = sqrt(dx_look*dx_look + dy_look*dy_look + dz_look*dz_look);

            dx_look /= dist;
            dy_look /= dist;
            dz_look /= dist;

            double pitch_angle = atan2(-dz_look, sqrt(dx_look*dx_look + dy_look*dy_look));
            double pitch_shake = multiOctaveNoise(elapsed + 800.0, 3.3, 0.01, 2);

            const double front_orient[4] = {
                -0.13, 0.0, 1.0, M_PI - (pitch_angle + 0.4 + pitch_shake) + M_PI/6.0
            };
            wb_supervisor_field_set_sf_rotation(main_rot, front_orient);

        } else if (current_view == 4 && vp_main && last_ball) {
            WbFieldRef ball_trans_field = wb_supervisor_node_get_field(last_ball, "translation");
            const double *ball_pos = wb_supervisor_field_get_sf_vec3f(ball_trans_field);

            WbFieldRef main_trans = wb_supervisor_node_get_field(vp_main, "position");
            WbFieldRef main_rot = wb_supervisor_node_get_field(vp_main, "orientation");

            const double ball_cam_pos[3] = {
                ball_pos[0] - 0.3,
                ball_pos[1] - 0.25,
                ball_pos[2] + 0.05
            };
            wb_supervisor_field_set_sf_vec3f(main_trans, ball_cam_pos);

            double dx_ball = ball_pos[0] - ball_cam_pos[0];
            double dy_ball = ball_pos[1] - ball_cam_pos[1];
            double dz_ball = ball_pos[2] - ball_cam_pos[2];
            double dist_ball = sqrt(dx_ball*dx_ball + dy_ball*dy_ball + dz_ball*dz_ball);

            dx_ball /= dist_ball;
            dy_ball /= dist_ball;
            dz_ball /= dist_ball;

            
            double yaw_angle = atan2(dy_ball, dx_ball);
            double pitch_angle = atan2(-dz_ball, sqrt(dx_ball*dx_ball + dy_ball*dy_ball));
            pitch_angle -= 1.0 * M_PI / 180.0;

            const double ball_orient[4] = {
                -sin(pitch_angle) * sin(yaw_angle),
                sin(pitch_angle) * cos(yaw_angle),
                cos(pitch_angle),
                yaw_angle
            };
            wb_supervisor_field_set_sf_rotation(main_rot, ball_orient);
        }

        if (elapsed - last_print >= 2.0) {
            const char* cam_names[] = {"Free", "Drone FPV", "Follow", "Front", "Ball Follow"};
            printf("Time: %.1fs | Pos: (%.2f, %.2f, %.2f) | Progress: %.1f%% | View: %s\n",
                   elapsed, x, y, z, progress * 100.0, cam_names[current_view]);
            last_print = elapsed;
        }

        if (drone2_finished) {
            printf("\n=== Both drones completed! Simulation finished ===\n");
            break;
        }
    }

    wb_robot_cleanup();
    return 0;
}