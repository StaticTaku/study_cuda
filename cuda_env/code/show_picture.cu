#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <cstddef>
#include <memory>
#include <new>
#include <matrix.hpp>
#include <fstream>

struct RGBA {
    unsigned char r, g, b, a; //赤, 緑, 青, 透過
    RGBA() = default;
    constexpr RGBA(const unsigned char r_, const unsigned char g_, const unsigned char b_, const unsigned char a_) :r(r_), g(g_), b(b_), a(a_) {}
};

using namespace cuda_Matrix;
int main() 
{
    std::string file_path;
    Matrix pic(28,28);
    constexpr std::size_t width{ 28 }, height{ 28 }; //幅と高さ
    while (true)
    {
        std::cout << "input a picture file path\n";
        std::cin  >>  file_path;
        load_data(file_path, pic);
        
        std::unique_ptr<RGBA[][width]> rgba(new(std::nothrow) RGBA[height][width]);
        if (!rgba) return -1;

        for (std::size_t row{}; row < height; ++row)
            for (std::size_t col{}; col < width; ++col) 
            {
                rgba[row][col].r = pic.elements[row*pic.width + col]*255;
                rgba[row][col].g = pic.elements[row*pic.width + col]*255;
                rgba[row][col].b = pic.elements[row*pic.width + col]*255;
                rgba[row][col].a = 255; //不透過
            }

        stbi_write_png("picture_display.png", static_cast<int>(width), static_cast<int>(height), static_cast<int>(sizeof(RGBA)), rgba.get(), 0);
        std::cout << "display complete\n";
    }
}