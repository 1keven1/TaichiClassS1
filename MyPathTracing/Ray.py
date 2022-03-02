import taichi as ti

vec3f = ti.types.vector(3, ti.f32)

Ray = ti.types.struct(
    ori=vec3f,
    dir=vec3f
)


@ti.func
def ray_at(ray, t):
    return ray.ori + t * ray.dir
