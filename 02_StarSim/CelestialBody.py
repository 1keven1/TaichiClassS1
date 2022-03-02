import taichi as ti

# 常量
G = 1
PI = 3.1415926


# 天体基类
@ti.data_oriented
class CelestialBody:
    def __init__(self, num, radius, density=1):
        self.num = num
        self.radius = radius
        self.volume = (4 / 3) * PI * ti.pow(radius, 3)
        self.mass = self.volume * density
        self.force = ti.Vector.field(2, ti.f32, shape=self.num)
        self.vel = ti.Vector.field(2, ti.f32, shape=self.num)
        self.pos = ti.Vector.field(2, ti.f32, shape=self.num)

    @ti.kernel
    def init(self, spawn_radius: ti.f32, init_speed: ti.f32):
        for n in range(self.num):
            radius = self.calculate_spawn_radius(n, self.num)
            offset_dir = ti.Vector([ti.cos(radius), ti.sin(radius)])
            self.pos[n] = ti.Vector([0.5, 0.5]) + offset_dir * spawn_radius
            self.vel[n] = ti.Vector([-offset_dir[1], offset_dir[0]]) * init_speed

    @ti.func
    def calculate_spawn_radius(self, i, num):
        return (i / ti.cast(num, ti.f32)) * 2 * PI

    @ti.kernel
    def calculate_force(self):
        self.clear_fource()
        for i in range(self.num):
            p = self.pos[i]
            for j in range(self.num):
                if j != i:
                    dir_vec = self.pos[j] - p
                    distance = dir_vec.norm(1e-2)
                    self.force[i] += G * self.mass * self.mass * dir_vec / distance ** 3

    @ti.func
    def clear_fource(self):
        for i in self.force:
            self.force[i] = ti.Vector([0, 0])

    @ti.kernel
    def update_pos(self, corona: ti.f32):
        for i in self.vel:
            self.vel[i] += corona * self.force[i] / self.mass
            self.pos[i] += corona * self.vel[i]

    def draw(self, gui: ti.GUI, color=0xffffff):
        gui.circles(self.pos.to_numpy(), radius=self.radius)

    @ti.func
    def get_num(self):
        return self.num

    @ti.func
    def get_pos(self):
        return self.pos

    @ti.func
    def get_mass(self):
        return self.mass


# 恒星子类
@ti.data_oriented
class Star(CelestialBody):
    def __init__(self, num, radius, density=1):
        super().__init__(num, radius, density)


@ti.data_oriented
class Planet(CelestialBody):
    @ti.kernel
    def init(self, spawn_radius: ti.f32, init_speed: ti.f32):
        for n in range(self.num):
            radius = self.calculate_spawn_radius(n, self.num)
            offset_dir = ti.Vector([ti.cos(radius), ti.sin(radius)])
            self.pos[n] = ti.Vector([0.5, 0.5]) + offset_dir * (spawn_radius + (ti.random() - 0.5) * 0.2)
            self.vel[n] = ti.Vector([-offset_dir[1], offset_dir[0]]) * init_speed

    @ti.kernel
    def calculate_force(self, stars: ti.template()):
        self.clear_fource()
        for i in range(self.num):
            p = self.pos[i]
            for j in range(self.num):
                if j != i:
                    dir_vec = self.pos[j] - p
                    distance = dir_vec.norm(1e-2)
                    self.force[i] += G * self.mass * self.mass * dir_vec / distance ** 3

            for j in range(stars.get_num()):
                dir_vec = stars.get_pos()[j] - p
                distance = dir_vec.norm(1e-2)
                self.force[i] += G * stars.get_mass() * self.mass * dir_vec / distance ** 3
