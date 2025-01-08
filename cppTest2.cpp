
#include <cstdio> 
#include <cstdlib> 
#include <fstream> 
#include "geometry.h" 
#include <cstring>
#include <math.h>

int imageWidth =49;
int imageHeight = 49;

Vec3<unsigned char> *frameBuffer = new Vec3<unsigned char>[imageWidth * imageHeight]; 

int main(){




char imageName[256] = "image2";
for (uint32_t m = 0; m < imageWidth * imageHeight; ++m) frameBuffer[m] = 
                    Vec3<unsigned char>(01, 25, 24);
    for(int x = 0; x < imageWidth; x++){
        for(int y=0;y<imageHeight;y++){
            frameBuffer[y * imageWidth + x].x = (unsigned char)(20 ); 
            frameBuffer[y * imageWidth + x].y = (unsigned char)(150 ); 
            frameBuffer[y * imageWidth + x].z = (unsigned char)( 100);

        }
    }

                     


  std::ofstream ofs2; 
    char buff[456];
    std::cerr<<"Saving image "<<imageName<<std::endl;
    // strcpy(imageName, "image2");
    sprintf(buff, "images/%s.ppm",imageName );
    ofs2.open(buff);            
    ofs2 << "P6\n" << imageWidth << " " << imageHeight << "\n255\n";
    ofs2.write((char*)frameBuffer, imageWidth * imageHeight * 3); 
    ofs2.close();

}