import pywavefront


def run():
    scene = pywavefront.Wavefront('truck.obj', parse=True)
    scene.vertices

if __name__ == '__main__':
    run()
