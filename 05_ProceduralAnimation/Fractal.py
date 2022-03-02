import taichi as ti
import ShaderFunction as sf

ti.init(arch=ti.cuda)

res = 512
pixel = ti.Vector.field(3, ti.f32, (res, res))


@ti.func
def draw_box(uv):
    uv = ti.sin(uv * 3.14)
    mask = uv[0] * uv[1]
    mask = ti.pow(mask, 0.5)
    return mask


@ti.kernel
def render(t: ti.f32):
    for i, j in pixel:
        pixel[i, j] = ti.Vector([1.0, 0.0, 0.0])

        final_color = ti.Vector([0.0, 0.0, 0.0])
        corona = 5
        for k in range(corona):
            # Tile
            big_num = 4 * 2 ** k
            ori_uv = ti.Vector([i, j]) / res + t * 0.3
            uv = sf.fract(ori_uv * big_num)

            mask = draw_box(uv)

            # random
            block = ti.floor(ori_uv * big_num)
            random = sf.fract(ti.sin(float(block[0] * 8 + block[1] * 17 + t * 0.03)) * 111)

            weight = corona - k
            color1 = ti.Vector([0.00, 0.0, 0.1])
            color2 = ti.Vector([1, 0.6, 0.6])
            final_color += sf.lerp(color1, color2, mask) * weight * random
            # final_color += random
        final_color /= float(corona) * 1.7

        pixel[i, j] = final_color


gui = ti.GUI("fracal", res=(res, res))

t = 0
while gui.running:
    t += 0.001
    render(t)
    gui.set_image(pixel)
    gui.show()
