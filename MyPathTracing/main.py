import taichi as ti
import numpy as np
from Scene import SceneList, Sphere
from Camera import Camera
from FunctionLib import random_unit_vector, reflect, refract, reflectance

ti.init(arch=ti.cuda)

res = 900
max_depth = 50
p_rr = 0.8

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res, res))


@ti.kernel
def render(frame: ti.int32):
    for i, j in pixels:
        u = (i + ti.random()) / float(res)
        v = (j + ti.random()) / float(res)

        # 摄像机发出射线
        color = ti.Vector([0.0, 0.0, 0.0])
        ray = camera.shoot_ray(u, v)
        color += ray_color(ray)

        # 根据帧数加权混合颜色
        if frame == 1:
            pixels[i, j] += color
        else:
            inverse_frame = 1 / frame
            pixels[i, j] *= 1 - inverse_frame
            pixels[i, j] += color * inverse_frame


@ti.func
def ray_color(ray):
    final_color = ti.Vector([1.0, 1.0, 1.0])
    final_brightness = 0

    for n in range(max_depth):
        if ti.random() > p_rr:
            break
        is_hit, hit_point, hit_point_normal, front_face, material, color = scene.hit(ray)
        if is_hit:
            # 灯
            if material == 0:
                final_color *= color * 10
                final_brightness = 1
                break
            # 漫反射
            elif material == 1:
                target = hit_point + hit_point_normal * 1
                target += random_unit_vector()
                ray.ori = hit_point
                ray.dir = target - hit_point
                final_color *= color
            # 金属
            elif material == 2 or material == 4:
                fuzz = 0.0
                if material == 4:
                    fuzz = 0.3
                ray.dir = reflect(ray.dir.normalized(), hit_point_normal)
                ray.dir += fuzz * random_unit_vector()
                ray.ori = hit_point
                if ray.dir.dot(hit_point_normal) <= 0:
                    break
                else:
                    final_color *= color
            # 玻璃
            elif material == 3:
                IOR = 1.5
                if front_face:
                    IOR = 1 / IOR
                cos_theta = min(-ray.dir.normalized().dot(hit_point_normal), 1.0)
                sin_theta = ti.sqrt(1 - cos_theta * cos_theta)
                # total internal reflection
                if IOR * sin_theta > 1.0 or reflectance(cos_theta, IOR) > ti.random():
                    ray.dir = reflect(ray.dir.normalized(), hit_point_normal)
                else:
                    ray.dir = refract(ray.dir.normalized(), hit_point_normal, IOR)
                ray.ori = hit_point
                final_color *= color

            final_color /= p_rr

    return final_color * final_brightness

# Gamma矫正
@ti.kernel
def gamma():
    for i, j in pixels:
        pixels[i, j] = ti.pow(pixels[i, j], 1/2.2)


scene = SceneList()
# Light source
scene.add(Sphere(center=ti.Vector([0, 5.4, -1]), radius=3.0, material=0, color=ti.Vector([1.0, 1.0, 1.0])))
# Ground
scene.add(Sphere(center=ti.Vector([0, -100.5, -1]), radius=100.0, material=1, color=ti.Vector([0.8, 0.8, 0.8])))
# ceiling
scene.add(Sphere(center=ti.Vector([0, 102.5, -1]), radius=100.0, material=1, color=ti.Vector([0.8, 0.8, 0.8])))
# back wall
scene.add(Sphere(center=ti.Vector([0, 1, 101]), radius=100.0, material=1, color=ti.Vector([0.8, 0.8, 0.8])))
# right wall
scene.add(Sphere(center=ti.Vector([-101.5, 0, -1]), radius=100.0, material=1, color=ti.Vector([0.6, 0.0, 0.0])))
# left wall
scene.add(Sphere(center=ti.Vector([101.5, 0, -1]), radius=100.0, material=1, color=ti.Vector([0.0, 0.6, 0.0])))

# Diffuse ball
scene.add(Sphere(center=ti.Vector([0, -0.2, -1.5]), radius=0.3, material=1, color=ti.Vector([0.8, 0.3, 0.3])))
# Metal ball
scene.add(Sphere(center=ti.Vector([-0.7, 0.2, -0.7]), radius=0.7, material=2, color=ti.Vector([0.6, 0.8, 0.8])))
# Glass ball
scene.add(Sphere(center=ti.Vector([0.7, 0, -0.5]), radius=0.5, material=3, color=ti.Vector([1.0, 1.0, 1.0])))
# Metal ball-2
scene.add(Sphere(center=ti.Vector([0.6, -0.3, -2.0]), radius=0.2, material=4, color=ti.Vector([0.8, 0.6, 0.2])))

camera = Camera(ti.Vector([0.0, 1.0, -5.0]), ti.Vector([0.0, 1.0, 0.0]), ti.Vector([0.0, 1.0, 0.0]), 60, 1)

gui = ti.GUI(name="Path Tracing", res=res)

frame = 0
while gui.running:
    frame += 1
    render(frame)
    gui.set_image(np.power(pixels.to_numpy(), 1 / 2.2))
    gui.show()


