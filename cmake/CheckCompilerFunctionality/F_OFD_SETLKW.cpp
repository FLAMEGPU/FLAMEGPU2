#ifdef __linux__
#include <fcntl.h>
#endif  // __linux__

int main (int argc, char * argv[]) {
    #ifndef F_OFD_SETLKW
    #error F_OFD_SETLKW is not defined; try building with -D_FILE_OFFSET_BITS=64
    #endif  // F_OFD_SETLKW
    return 0;
}