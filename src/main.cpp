#define TINYGLTF_IMPLEMENTATION
#include <tiny_gltf.h>
#undef TINYGLTF_IMPLEMENTATION

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#undef STB_IMAGE_IMPLEMENTATION

#ifdef _MSC_VER
# define STBI_MSC_SECURE_CRT
#endif
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#undef STB_IMAGE_WRITE_IMPLEMENTATION

#include "Program.h"

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cout << "Usage: PathTracer.exe <pathToScene> <pointLightsToGenerate>" << std::endl;
        return -1;
    }

    std::string scene           = std::string(argv[1]);
    uint32_t    pointLightCount = std::atoi(argv[2]);

    Program app(scene, pointLightCount);
    app.mainLoop();
    return 0;
}
