import taichi as ti

ti.init(arch=ti.gpu)

res = 720
pixel = ti.Vector.field(3, ti.f32, shape=(res, res))


@ti.func
def my_power(z):
    return ti.Vector([z[0] ** 2 - z[1] ** 2, 2 * z[0] * z[1]])


@ti.kernel
def calculate(t: ti.f32):
    for i, j in pixel:
        z = (ti.Vector([i / res, j / res]) - 0.5) * 4
        c = ti.Vector([-0.8, 0.156 * ti.sin(t) * 2])
        iteration = 0
        while iteration < 50 and z.norm() < 20:
            z = my_power(z) + c
            iteration += 1
        pixel[i, j] = ti.Vector([1, 0.4, 0.3]) * iteration * 0.07


gui = ti.GUI("Julia Set", res=(res, res))

time = 1

while gui.running:
    time += 0.03
    calculate(time)
    gui.set_image(pixel)
    gui.show()
