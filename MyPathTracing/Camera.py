import taichi as ti
from Ray import Ray
PI = 3.1415926

@ti.data_oriented
class Camera:
    def __init__(self, pos, lookat, up, fov, ratio):
        self.pos = pos
        self.lookat = lookat
        self.up = up
        self.fov = fov
        self.ratio = ratio

        # 需要计算的属性
        self.lower_left_corner = ti.Vector([0.0, 0.0, 0.0])
        self.vertical = ti.Vector([0.0, 0.0, 0.0])
        self.horizontal = ti.Vector([0.0, 0.0, 0.0])
        self.initialize()

    # 计算一些自己的属性
    def initialize(self):
        theta = (self.fov / 180.0) * PI
        half_height = ti.tan(theta / 2.0)
        half_width = half_height * self.ratio
        w = (self.pos - self.lookat).normalized()
        u = (self.up.cross(w)).normalized()
        v = w.cross(u)
        self.lower_left_corner = self.pos - u * half_width - v * half_height - w
        self.vertical = u * half_width * 2
        self.horizontal = v * half_height * 2
        print(self.pos, self.lower_left_corner, self.vertical, self.horizontal)

    # 向给定UV方向射出射线
    @ti.func
    def shoot_ray(self, u, v):
        origin = self.pos
        direction = self.lower_left_corner + u * self.vertical + v * self.horizontal - origin
        ray = Ray(ori=origin, dir=direction)
        return ray
