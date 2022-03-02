import taichi as ti

ti.init(arch=ti.gpu)

res = 512
pixel = ti.field(ti.f32, shape=(res, res))


@ti.func
def my_power(z):
    return ti.Vector([z[0] ** 2 - z[1] ** 2, 2 * z[0] * z[1]])


@ti.kernel
def calculate(t: ti.f32):
    for i, j in pixel:
        z = (ti.Vector([i / res, j / res]) - 0.5) * 2
        c = z
        iteration = 0
        while iteration < 50 and z.norm() < 10:
            z = my_power(z) + c
            iteration += 1
        pixel[i, j] = iteration * 0.02


gui = ti.GUI("Mandlebrot Set", res=(res, res))

t = 1
while gui.running:
    t += 0.03
    calculate(t)
    gui.set_image(pixel)
    gui.show()
