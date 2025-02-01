#include <iostream>
#include "parser.h"
#include "ppm.h"

typedef unsigned char RGB[3];

struct Ray {
    parser::Vec3f origin;
    parser::Vec3f direction;
    int depth;
};

bool intersect_sphere(const Ray& ray, const parser::Sphere& sphere, const parser::Vec3f& sphere_center, float& t)
{
    parser::Vec3f oc = subtract(ray.origin, sphere_center);
    float a = dot(ray.direction, ray.direction);
    float b = 2.0f * dot(oc, ray.direction);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;
    float discriminant = b * b - 4 * a * c;

    if (discriminant < 0)
    {
        return false; 
    }
    else
    {
        float t0 = (-b - std::sqrt(discriminant)) / (2.0f * a);
        float t1 = (-b + std::sqrt(discriminant)) / (2.0f * a);
        t = (t0 > 0) ? t0 : t1; 
        return t > 0;
    }
}

bool intersect_triangle(const Ray& ray, const parser::Vec3f& v0, const parser::Vec3f& v1, const parser::Vec3f& v2, const parser::Vec3f& normal, float& t)
{
    const float EPSILON = 1e-7f;
    parser::Vec3f edge1 = subtract(v1, v0);
    parser::Vec3f edge2 = subtract(v2, v0);

    parser::Vec3f h = cross(ray.direction, edge2);
    float a = dot(edge1, h);

    if (fabs(a) < EPSILON)
    {
        return false; 
    }

    float f = 1.0f / a;
    parser::Vec3f s = subtract(ray.origin, v0);
    float u = f * dot(s, h);

    if (u < 0.0f || u > 1.0f)
    {
        return false;
    }

    parser::Vec3f q = cross(s, edge1);
    float v = f * dot(ray.direction, q);

    if (v < 0.0f || u + v > 1.0f)
    {
        return false;
    }


    t = f * dot(edge2, q);

    return t > EPSILON; 
}

parser::Vec3f compute_color(const Ray& ray, const parser::Scene& scene);

parser::Vec3f apply_shading(const Ray& ray, const parser::Material& material, const parser::Vec3f& intersection_point, const parser::Vec3f& normal, const parser::Scene& scene) {
    
    parser::Vec3f color = {
        static_cast<float>(scene.ambient_light.x * material.ambient.x),
        static_cast<float>(scene.ambient_light.y * material.ambient.y),
        static_cast<float>(scene.ambient_light.z * material.ambient.z)
    };

    const parser::Vec3f& view_dir = get_normal(subtract(ray.origin, intersection_point));
    if (material.is_mirror) {
        const parser::Vec3f& reflection_origin = add(intersection_point, mult(normal, scene.shadow_ray_epsilon));
        const parser::Vec3f& reflection_dir = get_normal(add(mult(view_dir, -1), mult(normal, 2 * dot(normal, view_dir))));
        Ray reflection_ray = {
            reflection_origin,
            reflection_dir,
            ray.depth + 1
        };
        const auto& k_m = material.mirror;
        const parser::Vec3f& reflected_color = compute_color(reflection_ray, scene);
        color.x += reflected_color.x * k_m.x;
        color.y += reflected_color.y * k_m.y;
        color.z += reflected_color.z * k_m.z;
    }

    for (const auto& light : scene.point_lights) {
        parser::Vec3f light_dir = subtract(light.position, intersection_point);
        float light_distance = std::sqrt(dot(light_dir, light_dir));
        light_dir = get_normal(light_dir);

        Ray shadow_ray;
        shadow_ray.origin = add(intersection_point, mult(normal, scene.shadow_ray_epsilon));
        shadow_ray.direction = light_dir;

        bool in_shadow = false;

        for (const auto& sphere : scene.spheres) {
            parser::Vec3f sphere_center = sphere.center;
            float t;
            if (intersect_sphere(shadow_ray, sphere, sphere_center, t) && t < light_distance) {
                in_shadow = true;
                break;
            }
        }

        for (const auto& triangle : scene.triangles) {
            if (in_shadow) break;
            float t;
            //BACKFACE CULLING
            if (dot(triangle.normal, shadow_ray.direction) <= 0 && intersect_triangle(shadow_ray, triangle.v0, triangle.v1, triangle.v2, triangle.normal, t) && t < light_distance) {
                in_shadow = true;
                break;
            }
        }

        for (const auto& mesh : scene.meshes) {
            if (in_shadow) break;
            for (const auto& face : mesh.faces) {
                float t;
                //BACKFACE CULLING
                if (dot(face.normal, shadow_ray.direction) <= 0 && intersect_triangle(shadow_ray, face.v0, face.v1, face.v2, face.normal, t) && t < light_distance) {
                    in_shadow = true;
                    break;
                }
            }
        }

        if (!in_shadow) {
            if (dot(light_dir, normal) > 0) {
                // DIFFUSE
                const auto& k_d = material.diffuse;
                const float cos_theta = std::max(0.0f, dot(light_dir, normal));
                const float light_dist_squared = light_distance * light_distance;

                color.x += k_d.x * cos_theta * light.intensity.x / light_dist_squared;
                color.y += k_d.y * cos_theta * light.intensity.y / light_dist_squared;
                color.z += k_d.z * cos_theta * light.intensity.z / light_dist_squared;

                // SPECULAR
                parser::Vec3f half_vector = get_normal(add(light_dir, view_dir));
                const float spec_angle = std::max(0.0f, dot(normal, half_vector));
                const float cos_alpha = std::pow(spec_angle, material.phong_exponent);
                const auto& k_s = material.specular;

                color.x += k_s.x * cos_alpha * light.intensity.x / light_dist_squared;
                color.y += k_s.y * cos_alpha * light.intensity.y / light_dist_squared;
                color.z += k_s.z * cos_alpha * light.intensity.z / light_dist_squared;
            }
        }
    }

    return color;
}

parser::Vec3f compute_color(const Ray& ray, const parser::Scene& scene){

    // MAX DEPTH EXCEDED
    if (ray.depth > scene.max_recursion_depth) {
        return {
            static_cast<float>(0),
            static_cast<float>(0),
            static_cast<float>(0)
        };
    }
    
    // FIND CLOSEST HIT
    float tmin = std::numeric_limits<float>::max();
    parser::Material tmin_material;
    parser::Vec3f tmin_intersection_point;
    parser::Vec3f tmin_normal;
    bool hit = false;

    for (const auto& sphere : scene.spheres) {
        parser::Vec3f sphere_center = sphere.center;
        float t;
        if (intersect_sphere(ray, sphere, sphere_center, t) && t < tmin) {
            const parser::Vec3f& intersection_point = add(ray.origin, mult(ray.direction, t));
            const parser::Vec3f& normal = get_normal(subtract(intersection_point, sphere_center)); 
            int mat_id = sphere.material_id - 1; 
            tmin = t;
            tmin_material = scene.materials[mat_id];
            tmin_intersection_point = intersection_point;
            tmin_normal = normal;
            hit = true;
        }
    }
    for (const auto& mesh : scene.meshes) {
        int mat_id = mesh.material_id - 1;
        for (const auto& face : mesh.faces) {
            float t;
            //BACKFACE CULLING
            if (dot(face.normal, ray.direction) <= 0 && intersect_triangle(ray, face.v0, face.v1, face.v2, face.normal, t) && t < tmin) {
                const parser::Vec3f& intersection_point = add(ray.origin, mult(ray.direction, t));
                const parser::Vec3f& normal = face.normal;
                tmin = t;
                tmin_material = scene.materials[mat_id];
                tmin_intersection_point = intersection_point;
                tmin_normal = normal;
                hit = true;
            }
        }
    }
    for (const auto& triangle : scene.triangles) {
        int mat_id = triangle.material_id - 1;
        float t;
        //BACKFACE CULLING
        if (dot(triangle.normal, ray.direction) <= 0 && intersect_triangle(ray, triangle.v0, triangle.v1, triangle.v2, triangle.normal, t) && t < tmin) {
            const parser::Vec3f& intersection_point = add(ray.origin, mult(ray.direction, t));
            const parser::Vec3f& normal = triangle.normal;
            tmin = t;
            tmin_material = scene.materials[mat_id];
            tmin_intersection_point = intersection_point;
            tmin_normal = normal;
            hit = true;
        }
    }
    if (hit) {
        return apply_shading(ray, tmin_material, tmin_intersection_point, tmin_normal, scene);
    }
    else{
        // NO INTERSECTION FOR THE PRIMARY RAY
        if (ray.depth == 0) {
            return {
                static_cast<float>(scene.background_color.x),
                static_cast<float>(scene.background_color.y),
                static_cast<float>(scene.background_color.z)
            };
        }
        //AVOID REPEATED ADDITION OF THE BACKGROUND COLOR FOR REFLECTED RAYS
        else{
            return {
                static_cast<float>(0),
                static_cast<float>(0),
                static_cast<float>(0)
            };
        }
    }
}

template <typename T>
T clamp(T value, T min, T max) {
    return (value < min) ? min : (value > max) ? max : value;
}


struct ThreadData {
    int start_row;
    int end_row;
    int image_width;
    unsigned char* image;
    const parser::Scene* scene;
    const parser::Camera* camera;
};


void* render_section(void* arg) {
    ThreadData* data = static_cast<ThreadData*>(arg);
    int image_width = data->image_width;

    const parser::Vec3f& camera_position = data->camera->position;
    parser::Vec3f gaze = get_normal(data->camera->gaze);
    parser::Vec3f up = get_normal(data->camera->up);
    float near_distance = data->camera->near_distance;
    parser::Vec4f near_plane = data->camera->near_plane;

    parser::Vec3f right = get_normal(cross(gaze, up));
    parser::Vec3f m = add(camera_position, mult(gaze, near_distance));
    parser::Vec3f q = add(m, add(mult(up, near_plane.w), mult(right, near_plane.x)));

    int idx = data->start_row * image_width * 3;

    for (int j = data->start_row; j < data->end_row; j++) {
        for (int i = 0; i < image_width; i++) {
            float s_u = (i + 0.5f) * (near_plane.y - near_plane.x) / image_width;
            float s_v = (j + 0.5f) * (near_plane.w - near_plane.z) / data->camera->image_height;

            parser::Vec3f s = add(q, add(mult(right, s_u), mult(up, -s_v)));

            Ray ray;
            ray.origin = camera_position;
            ray.direction = get_normal(subtract(s, ray.origin));
            ray.depth = 0;

            parser::Vec3f color = compute_color(ray, *data->scene);

            data->image[idx++] = static_cast<unsigned char>(clamp(color.x, 0.0f, 255.0f));
            data->image[idx++] = static_cast<unsigned char>(clamp(color.y, 0.0f, 255.0f));
            data->image[idx++] = static_cast<unsigned char>(clamp(color.z, 0.0f, 255.0f));
        }
    }

    return nullptr;
}

int main(int argc, char* argv[]) {

    parser::Scene scene;
    scene.loadFromXml(argv[1]);

    const int num_threads = 8; 
    
    for (const auto& camera : scene.cameras) {
        int image_width = camera.image_width;
        int image_height = camera.image_height;
        unsigned char* image_created = new unsigned char[image_width * image_height * 3];

        std::vector<pthread_t> threads(num_threads);
        std::vector<ThreadData> thread_data(num_threads);

        int rows_per_thread = image_height / num_threads;

        for (int i = 0; i < num_threads; i++) {
            thread_data[i] = {
                i * rows_per_thread,
                (i == num_threads - 1) ? image_height : (i + 1) * rows_per_thread,
                image_width,
                image_created,
                &scene,
                &camera
            };
            pthread_create(&threads[i], nullptr, render_section, &thread_data[i]);
        }

        for (auto& thread : threads) {
            pthread_join(thread, nullptr);
        }

        write_ppm(camera.image_name.c_str(), image_created, image_width, image_height);

        delete[] image_created;
    }

    return 0;
}
