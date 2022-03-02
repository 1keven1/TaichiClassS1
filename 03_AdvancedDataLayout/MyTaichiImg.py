import taichi as ti

ti.init(ti.gpu, packed=True, debug=True)

x = ti.field(ti.f32)
block1 = ti.root.pointer(ti.ij, 16)
block1.dense(ti.ij, 40).place(x)

res = 16 * 40
pixel = ti.field(ti.f32, (res, res))


@ti.kernel
def taichi_img(t: ti.f32):
    for i, j in ti.ndrange(res, res):
        pos = ti.Vector([i, j]) / res
        pos = ti.Matrix.rotation2d(t) @ (pos - 0.5) + 0.5
        if ti.taichi_logo(pos) == 0:
            x[i, j] = 1


@ti.kernel
def show_pointer():
    for i, j in ti.ndrange(res, res):
        block_index = ti.rescale_index(x, block1, [i, j])
        pixel[i, j] = ti.is_active(block1, block_index) * 0.2 + x[i, j]


gui = ti.GUI("Taichi Image", res=res)

t = 0.00
while gui.running:
    t += 0.02
    block1.deactivate_all()
    taichi_img(t)
    show_pointer()
    gui.set_image(pixel)
    gui.show()
