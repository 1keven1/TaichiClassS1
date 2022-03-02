import taichi as ti
from Ray import Ray, ray_at


# 球形物体
@ti.data_oriented
class Sphere:
    def __init__(self, center, radius, material, color):
        self.center = center
        self.radius = radius
        self.material = material
        self.color = color

    # 求交函数
    @ti.func
    def hit(self, ray, t_min=0.001, t_max=10e8):
        oc = ray.ori - self.center
        a = ray.dir.dot(ray.dir)
        b = 2.0 * oc.dot(ray.dir)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c
        is_hit = False
        front_face = False
        root = 0.0
        hit_point = ti.Vector([0.0, 0.0, 0.0])
        hit_point_normal = ti.Vector([0.0, 0.0, 0.0])
        if discriminant > 0:
            sqrtd = ti.sqrt(discriminant)
            root = (-b - sqrtd) / (2 * a)
            if root < t_min or root > t_max:
                root = (-b + sqrtd) / (2 * a)
                if root >= t_min and root <= t_max:
                    is_hit = True
            else:
                is_hit = True
        if is_hit:
            hit_point = ray_at(ray, root)
            hit_point_normal = (hit_point - self.center) / self.radius
            # 检测是外面还是里面
            if ray.dir.dot(hit_point_normal) < 0:
                front_face = True
            else:
                hit_point_normal = -hit_point_normal
        return is_hit, root, hit_point, hit_point_normal, front_face, self.material, self.color


@ti.data_oriented
class SceneList:
    def __init__(self):
        self.list = []

    def add(self, obj):
        self.list.append(obj)

    def clear(self):
        self.list = []

    # 求交函数
    @ti.func
    def hit(self, ray, t_min=0.001, t_max=10e8):
        closest_t = t_max
        is_hit = False
        front_face = False
        hit_point = ti.Vector([0.0, 0.0, 0.0])
        hit_point_normal = ti.Vector([0.0, 0.0, 0.0])
        color = ti.Vector([0.0, 0.0, 0.0])
        material = 1
        for index in ti.static(range(len(self.list))):
            is_hit_tmp, root_tmp, hit_point_tmp, hit_point_normal_tmp, front_face_tmp, material_tmp, color_tmp = \
                self.list[index].hit(ray, t_min, closest_t)
            if is_hit_tmp:
                closest_t = root_tmp
                is_hit = is_hit_tmp
                hit_point = hit_point_tmp
                hit_point_normal = hit_point_normal_tmp
                front_face = front_face_tmp
                material = material_tmp
                color = color_tmp
        return is_hit, hit_point, hit_point_normal, front_face, material, color
