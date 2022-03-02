import taichi as ti
from CelestialBody import Star, Planet

ti.init(ti.cuda)

stars = Star(num=3, radius=7)
planets = Planet(100, 2)
stars.init(0.2, 10)
planets.init(0.3, 50)

gui = ti.GUI("Galaxy", (512, 512))
corona = 5e-5
while gui.running:
    stars.calculate_force()
    planets.calculate_force(stars)
    stars.update_pos(corona)
    planets.update_pos(corona)

    stars.draw(gui, 0xffffff)
    planets.draw(gui)
    gui.show()
