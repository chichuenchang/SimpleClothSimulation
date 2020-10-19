#include <iostream>

// These two files are downloaded from https://github.com/nothings/stb.
// The other 2 cpp files, stb_image.cpp and stb_image_write.cpp, are created by me.
// (Those 2 files are rather simple. Checkout the first 10 lines of comment in
//  the header files to see why the .cpp files are created.)
#include "stb_image.h"
#include "stb_image_write.h"

int main() {
    const char * filename = "statue.jpg";
    int image_height = 0, image_width = 0, num_channels = 0;
    unsigned char * image_data = stbi_load(filename, &image_width, &image_height, &num_channels, 0);

    // Do some simple checking.
    if (image_data == nullptr) {
        std::cerr << "Image reading failed." << std::endl;
        return 1;
    } else if (num_channels != 3 && num_channels != 4) {
        std::cerr << "The loaded image doesn't have RGB color components." << std::endl;
        std::cerr << "The loaded image has " << num_channels << " channels" << std::endl;
        return 1;
    } else {
        std::cout << "The image loaded has size " << image_width << "x" << image_height << std::endl;
    }

    // Now you can load this image as a texture using OpenGL calls.
    // glBindTexture(GL_TEXTURE_2D, ...);
    // glTexImage2D(....);
    // glBindTexture(GL_TEXTURE_2D, 0);
    //
    // If you see that the image loaded onto the screen is upside down,
    // you should consider calling the following function in stb_image:
    // stbi_set_flip_vertically_on_load(int flag_true_if_should_flip);

    // Demonstrates stb_image_write functions.
    // Writes the image to a different file.
    stbi_write_png("statue.png", image_width, image_height, num_channels, image_data,
                   image_width*num_channels);

    // stbi_load returns a piece of dynamically allocated memory.
    // As a good practice, the memory is released here.
    stbi_image_free(image_data);
    return 0;
}