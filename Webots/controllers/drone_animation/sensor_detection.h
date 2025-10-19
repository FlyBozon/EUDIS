#ifndef SENSOR_DETECTION_H
#define SENSOR_DETECTION_H

#include <webots/supervisor.h>
#include <math.h>
#include <stdbool.h>

#ifndef SENSOR_RANGE
#define SENSOR_RANGE 200.0
#endif

#ifndef DRONE2_ALTITUDE
#define DRONE2_ALTITUDE 7.5
#endif

typedef struct {
    WbNodeRef node;
    double position[3];
    bool active;
    double signal_strength;
} Sensor;

typedef struct {
    WbNodeRef zone_node;
    double center[3];
    double radius;
    bool visible;
} DetectionZone;

double calculate_signal_strength(double distance) {
    if (distance > SENSOR_RANGE) {
        return 0.0;
    }
    
    double normalized_dist = distance / SENSOR_RANGE;
    return 1.0 - (normalized_dist * normalized_dist);
}

double distance_3d(const double p1[3], const double p2[3]) {
    double dx = p2[0] - p1[0];
    double dy = p2[1] - p1[1];
    double dz = p2[2] - p1[2];
    return sqrt(dx*dx + dy*dy + dz*dz);
}

void estimate_detection_zone(const Sensor *sensor1, const Sensor *sensor2,
                            double drone_pos[3], double *zone_radius) {
    double sensor_distance = distance_3d(sensor1->position, sensor2->position);
    double dist1 = SENSOR_RANGE * sqrt(1.0 - sensor1->signal_strength);
    double dist2 = SENSOR_RANGE * sqrt(1.0 - sensor2->signal_strength);

    double dx = sensor2->position[0] - sensor1->position[0];
    double dy = sensor2->position[1] - sensor1->position[1];

    
    if (sensor_distance < 0.1) {
        drone_pos[0] = sensor1->position[0];
        drone_pos[1] = sensor1->position[1];
        drone_pos[2] = dist1;
        *zone_radius = 50.0;
        return;
    }

    
    double a = (dist1*dist1 - dist2*dist2 + sensor_distance*sensor_distance) / (2.0 * sensor_distance);

    double h_squared = dist1*dist1 - a*a;
    double h = (h_squared > 0) ? sqrt(h_squared) : 0;

    drone_pos[0] = sensor1->position[0] + a * dx / sensor_distance;
    drone_pos[1] = sensor1->position[1] + a * dy / sensor_distance;

    
    drone_pos[2] = DRONE2_ALTITUDE;
    *zone_radius = h + 10.0;
    if (*zone_radius < 5.0) *zone_radius = 5.0;
}

#endif 