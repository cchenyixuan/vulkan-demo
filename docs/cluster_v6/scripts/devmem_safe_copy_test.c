/* Exhaustive check of the shim's aligned copy / move / fill paths against a byte reference. */
#include "devmem_safe_copy.c"

static int failures = 0;

static void check(const unsigned char *actual, const unsigned char *expected, size_t length, const char *what,
                  int source_offset, int destination_offset, size_t size)
{
    for (size_t byte_index = 0; byte_index < length; ++byte_index) {
        if (actual[byte_index] != expected[byte_index]) {
            if (failures < 10)
                fprintf(stderr, "FAIL %s src+%d dst+%d size=%zu at byte %zu\n", what, source_offset,
                        destination_offset, size, byte_index);
            ++failures;
            return;
        }
    }
}

int main(void)
{
    enum { BUFFER = 640 };
    static unsigned char source[BUFFER], destination[BUFFER], expected[BUFFER], shared[BUFFER], scratch[BUFFER];
    unsigned long cases = 0;
    for (int source_offset = 0; source_offset < 16; ++source_offset)
        for (int destination_offset = 0; destination_offset < 16; ++destination_offset)
            for (size_t size = 0; size <= 300; ++size) {
                for (int byte_index = 0; byte_index < BUFFER; ++byte_index) {
                    source[byte_index] = (unsigned char)(byte_index * 7 + 3);
                    destination[byte_index] = expected[byte_index] = 0xAA;
                }
                for (size_t byte_index = 0; byte_index < size; ++byte_index)
                    expected[destination_offset + byte_index] = source[source_offset + byte_index];
                aligned_copy_forward(destination + destination_offset, source + source_offset, size);
                check(destination, expected, BUFFER, "forward", source_offset, destination_offset, size);

                for (int byte_index = 0; byte_index < BUFFER; ++byte_index) destination[byte_index] = 0xAA;
                aligned_copy_backward(destination + destination_offset, source + source_offset, size);
                check(destination, expected, BUFFER, "backward", source_offset, destination_offset, size);

                for (int byte_index = 0; byte_index < BUFFER; ++byte_index)
                    destination[byte_index] = expected[byte_index] = 0xAA;
                for (size_t byte_index = 0; byte_index < size; ++byte_index)
                    expected[destination_offset + byte_index] = (unsigned char)(0x5C + source_offset);
                aligned_fill(destination + destination_offset, 0x5C + source_offset, size);
                check(destination, expected, BUFFER, "fill", source_offset, destination_offset, size);

                /* overlapping move inside one buffer, both directions, through the shim's memmove dispatch */
                for (int shift = -40; shift <= 40; shift += 13) {
                    int from = 64 + source_offset, to = 64 + destination_offset + shift;
                    for (int byte_index = 0; byte_index < BUFFER; ++byte_index)
                        shared[byte_index] = expected[byte_index] = (unsigned char)(byte_index * 13 + 1);
                    for (size_t byte_index = 0; byte_index < size; ++byte_index)
                        scratch[byte_index] = expected[from + byte_index];
                    for (size_t byte_index = 0; byte_index < size; ++byte_index)
                        expected[to + byte_index] = scratch[byte_index];
                    if ((uintptr_t)(shared + to) > (uintptr_t)(shared + from)
                        && (uintptr_t)(shared + to) < (uintptr_t)(shared + from) + size)
                        aligned_copy_backward(shared + to, shared + from, size);
                    else
                        aligned_copy_forward(shared + to, shared + from, size);
                    check(shared, expected, BUFFER, "overlap", from, to, size);
                }
                cases += 4;
            }
    printf("devmem_safe_copy_test: %lu cases, %d failures\n", cases, failures);
    return failures != 0;
}
