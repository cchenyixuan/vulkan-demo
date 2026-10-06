/*
 * devmem_safe_copy.c - LD_PRELOAD shim for aarch64 hosts whose NVIDIA driver
 * maps GPU memory (BAR1, /dev/nvidiaN) as Device memory.
 *
 * Found on N32-H (Kunpeng-920, Kylin V10 kernel 4.19, A100-PCIE-40GB, driver
 * 535.104.12; E30 jobs 1550014 / 1550133 / 1550139): during
 * vkCmdBindDescriptorSets the Vulkan driver (libnvidia-eglcore) copies about
 * 184 bytes with glibc's memcpy into a 2 MiB /dev/nvidia0 mapping. glibc
 * 2.28's aarch64 memcpy aligns the SOURCE and issues 16-byte stores at
 * whatever destination alignment follows; on Device memory a 16-byte store
 * at an 8-byte aligned address raises SIGBUS (si_code 1 = BUS_ADRALN).
 *
 * The shim records every mmap of an NVIDIA GPU device node (character major
 * 195, minor below 254, i.e. /dev/nvidia0, /dev/nvidia1, ...; /dev/nvidiactl
 * and /dev/nvidia-modeset only with DEVMEM_SAFE_TRACK=all) and sends the
 * memcpy / memmove / memset calls that touch a recorded range through
 * naturally aligned 1-, 4- and 8-byte accesses. Every other call goes to
 * glibc unchanged, so host-to-host copies (transport staging) keep glibc's
 * speed. It changes no result: a copy is a copy.
 *
 * Build: gcc -O2 -fPIC -shared -fno-builtin -fno-tree-loop-distribute-patterns \
 *            -fno-tree-vectorize -o devmem_safe_copy.so devmem_safe_copy.c -ldl -lpthread
 * Use:   LD_PRELOAD=/path/devmem_safe_copy.so python ...
 * Prints one line to stderr at load and one at exit (recorded mappings and
 * how many calls took the aligned path).
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/sysmacros.h>
#include <sys/types.h>
#include <unistd.h>

#define NVIDIA_CHARACTER_MAJOR 195
#define NVIDIA_FIRST_CONTROL_MINOR 254
#define MAXIMUM_RECORDED_RANGES 4096

typedef struct {
    uintptr_t start;
    uintptr_t end;      /* exclusive; 0 marks a free slot */
} recorded_range;

static recorded_range recorded_ranges[MAXIMUM_RECORDED_RANGES];
static int recorded_range_count = 0;          /* slots in use, high-water */
static uintptr_t recorded_lowest_address = UINTPTR_MAX;
static uintptr_t recorded_highest_address = 0;
static pthread_mutex_t recorded_ranges_lock = PTHREAD_MUTEX_INITIALIZER;
static int track_all_nvidia_nodes = 0;

static unsigned long recorded_mapping_total = 0;
static unsigned long aligned_memcpy_calls = 0;
static unsigned long aligned_memmove_calls = 0;
static unsigned long aligned_memset_calls = 0;
static unsigned long aligned_bytes = 0;

static void *(*glibc_memcpy)(void *, const void *, size_t);
static void *(*glibc_memmove)(void *, const void *, size_t);
static void *(*glibc_memset)(void *, int, size_t);
static void *(*glibc_mmap)(void *, size_t, int, int, int, off_t);
static int (*glibc_munmap)(void *, size_t);

/* ---------- aligned copy and fill (every access naturally aligned) ---------- */

static void aligned_copy_forward(unsigned char *destination, const unsigned char *source, size_t size)
{
    uintptr_t relative_alignment = (uintptr_t)destination ^ (uintptr_t)source;
    if ((relative_alignment & 7) == 0) {
        while (size > 0 && ((uintptr_t)destination & 7) != 0) {
            *(volatile unsigned char *)destination++ = *(const volatile unsigned char *)source++;
            --size;
        }
        while (size >= 8) {
            *(volatile uint64_t *)destination = *(const volatile uint64_t *)source;
            destination += 8;
            source += 8;
            size -= 8;
        }
    } else if ((relative_alignment & 3) == 0) {
        while (size > 0 && ((uintptr_t)destination & 3) != 0) {
            *(volatile unsigned char *)destination++ = *(const volatile unsigned char *)source++;
            --size;
        }
        while (size >= 4) {
            *(volatile uint32_t *)destination = *(const volatile uint32_t *)source;
            destination += 4;
            source += 4;
            size -= 4;
        }
    }
    while (size > 0) {
        *(volatile unsigned char *)destination++ = *(const volatile unsigned char *)source++;
        --size;
    }
}

static void aligned_copy_backward(unsigned char *destination, const unsigned char *source, size_t size)
{
    unsigned char *destination_end = destination + size;
    const unsigned char *source_end = source + size;
    uintptr_t relative_alignment = (uintptr_t)destination_end ^ (uintptr_t)source_end;
    if ((relative_alignment & 7) == 0) {
        while (size > 0 && ((uintptr_t)destination_end & 7) != 0) {
            *(volatile unsigned char *)--destination_end = *(const volatile unsigned char *)--source_end;
            --size;
        }
        while (size >= 8) {
            destination_end -= 8;
            source_end -= 8;
            *(volatile uint64_t *)destination_end = *(const volatile uint64_t *)source_end;
            size -= 8;
        }
    } else if ((relative_alignment & 3) == 0) {
        while (size > 0 && ((uintptr_t)destination_end & 3) != 0) {
            *(volatile unsigned char *)--destination_end = *(const volatile unsigned char *)--source_end;
            --size;
        }
        while (size >= 4) {
            destination_end -= 4;
            source_end -= 4;
            *(volatile uint32_t *)destination_end = *(const volatile uint32_t *)source_end;
            size -= 4;
        }
    }
    while (size > 0) {
        *(volatile unsigned char *)--destination_end = *(const volatile unsigned char *)--source_end;
        --size;
    }
}

static void aligned_fill(unsigned char *destination, int value, size_t size)
{
    unsigned char byte_value = (unsigned char)value;
    uint64_t word_value = 0x0101010101010101ULL * byte_value;
    while (size > 0 && ((uintptr_t)destination & 7) != 0) {
        *(volatile unsigned char *)destination++ = byte_value;
        --size;
    }
    while (size >= 8) {
        *(volatile uint64_t *)destination = word_value;
        destination += 8;
        size -= 8;
    }
    while (size > 0) {
        *(volatile unsigned char *)destination++ = byte_value;
        --size;
    }
}

/* ---------- recorded device ranges ---------- */

static int touches_recorded_range(const void *address, size_t size)
{
    if (size == 0)
        return 0;
    uintptr_t first = (uintptr_t)address;
    uintptr_t last = first + size;      /* exclusive */
    if (last <= __atomic_load_n(&recorded_lowest_address, __ATOMIC_RELAXED)
        || first >= __atomic_load_n(&recorded_highest_address, __ATOMIC_RELAXED))
        return 0;
    int count = __atomic_load_n(&recorded_range_count, __ATOMIC_ACQUIRE);
    for (int slot_index = 0; slot_index < count; ++slot_index) {
        uintptr_t end = __atomic_load_n(&recorded_ranges[slot_index].end, __ATOMIC_ACQUIRE);
        uintptr_t start = __atomic_load_n(&recorded_ranges[slot_index].start, __ATOMIC_RELAXED);
        if (first < end && last > start)
            return 1;
    }
    return 0;
}

static int is_tracked_nvidia_node(int descriptor)
{
    struct stat status;
    if (descriptor < 0 || fstat(descriptor, &status) != 0 || !S_ISCHR(status.st_mode))
        return 0;
    if (major(status.st_rdev) != NVIDIA_CHARACTER_MAJOR)
        return 0;
    return track_all_nvidia_nodes || minor(status.st_rdev) < NVIDIA_FIRST_CONTROL_MINOR;
}

static void record_range(uintptr_t start, size_t size)
{
    uintptr_t end = start + size;
    pthread_mutex_lock(&recorded_ranges_lock);
    int chosen_slot = -1;
    for (int slot_index = 0; slot_index < recorded_range_count; ++slot_index) {
        if (recorded_ranges[slot_index].end == 0) {
            chosen_slot = slot_index;
            break;
        }
    }
    if (chosen_slot < 0 && recorded_range_count < MAXIMUM_RECORDED_RANGES)
        chosen_slot = recorded_range_count;
    if (chosen_slot >= 0) {
        __atomic_store_n(&recorded_ranges[chosen_slot].start, start, __ATOMIC_RELEASE);
        __atomic_store_n(&recorded_ranges[chosen_slot].end, end, __ATOMIC_RELEASE);
        if (chosen_slot == recorded_range_count)
            __atomic_store_n(&recorded_range_count, recorded_range_count + 1, __ATOMIC_RELEASE);
        if (start < recorded_lowest_address)
            __atomic_store_n(&recorded_lowest_address, start, __ATOMIC_RELAXED);
        if (end > recorded_highest_address)
            __atomic_store_n(&recorded_highest_address, end, __ATOMIC_RELAXED);
        ++recorded_mapping_total;
    } else {
        static const char message[] = "[devmem_safe] WARNING: range table full, mapping not recorded\n";
        write(2, message, sizeof message - 1);
    }
    pthread_mutex_unlock(&recorded_ranges_lock);
}

static void forget_range(uintptr_t start, size_t size)
{
    uintptr_t end = start + size;
    pthread_mutex_lock(&recorded_ranges_lock);
    for (int slot_index = 0; slot_index < recorded_range_count; ++slot_index) {
        recorded_range *range = &recorded_ranges[slot_index];
        if (range->end == 0 || end <= range->start || start >= range->end)
            continue;
        if (start <= range->start && end >= range->end) {          /* whole range unmapped */
            __atomic_store_n(&range->end, 0, __ATOMIC_RELEASE);
            __atomic_store_n(&range->start, 0, __ATOMIC_RELEASE);
        } else if (start <= range->start) {                        /* head unmapped */
            __atomic_store_n(&range->start, end, __ATOMIC_RELEASE);
        } else if (end >= range->end) {                            /* tail unmapped */
            __atomic_store_n(&range->end, start, __ATOMIC_RELEASE);
        }                                                          /* middle hole: keep it whole */
    }
    pthread_mutex_unlock(&recorded_ranges_lock);
}

/* ---------- interposed functions ---------- */

void *memcpy(void *destination, const void *source, size_t size)
{
    if (glibc_memcpy != NULL && !touches_recorded_range(destination, size)
        && !touches_recorded_range(source, size))
        return glibc_memcpy(destination, source, size);
    if (glibc_memcpy != NULL) {
        __atomic_add_fetch(&aligned_memcpy_calls, 1, __ATOMIC_RELAXED);
        __atomic_add_fetch(&aligned_bytes, size, __ATOMIC_RELAXED);
    }
    aligned_copy_forward(destination, source, size);
    return destination;
}

void *memmove(void *destination, const void *source, size_t size)
{
    if (glibc_memmove != NULL && !touches_recorded_range(destination, size)
        && !touches_recorded_range(source, size))
        return glibc_memmove(destination, source, size);
    if (glibc_memmove != NULL) {
        __atomic_add_fetch(&aligned_memmove_calls, 1, __ATOMIC_RELAXED);
        __atomic_add_fetch(&aligned_bytes, size, __ATOMIC_RELAXED);
    }
    if ((uintptr_t)destination > (uintptr_t)source && (uintptr_t)destination < (uintptr_t)source + size)
        aligned_copy_backward(destination, source, size);
    else
        aligned_copy_forward(destination, source, size);
    return destination;
}

void *memset(void *destination, int value, size_t size)
{
    if (glibc_memset != NULL && !touches_recorded_range(destination, size))
        return glibc_memset(destination, value, size);
    if (glibc_memset != NULL) {
        __atomic_add_fetch(&aligned_memset_calls, 1, __ATOMIC_RELAXED);
        __atomic_add_fetch(&aligned_bytes, size, __ATOMIC_RELAXED);
    }
    aligned_fill(destination, value, size);
    return destination;
}

void *__memcpy_chk(void *destination, const void *source, size_t size, size_t destination_size)
{
    if (destination_size < size)
        abort();
    return memcpy(destination, source, size);
}

void *__memmove_chk(void *destination, const void *source, size_t size, size_t destination_size)
{
    if (destination_size < size)
        abort();
    return memmove(destination, source, size);
}

void *__memset_chk(void *destination, int value, size_t size, size_t destination_size)
{
    if (destination_size < size)
        abort();
    return memset(destination, value, size);
}

static void *mapping_with_record(void *address, size_t size, int protection, int flags, int descriptor, off_t offset)
{
    void *result = glibc_mmap != NULL
        ? glibc_mmap(address, size, protection, flags, descriptor, offset)
        : (void *)syscall(SYS_mmap, address, size, protection, flags, descriptor, offset);
    if (result != MAP_FAILED && is_tracked_nvidia_node(descriptor))
        record_range((uintptr_t)result, size);
    return result;
}

void *mmap(void *address, size_t size, int protection, int flags, int descriptor, off_t offset)
{
    return mapping_with_record(address, size, protection, flags, descriptor, offset);
}

void *mmap64(void *address, size_t size, int protection, int flags, int descriptor, off64_t offset)
{
    return mapping_with_record(address, size, protection, flags, descriptor, (off_t)offset);
}

int munmap(void *address, size_t size)
{
    int result = glibc_munmap != NULL ? glibc_munmap(address, size) : (int)syscall(SYS_munmap, address, size);
    if (result == 0)
        forget_range((uintptr_t)address, size);
    return result;
}

/* ---------- load / exit ---------- */

__attribute__((constructor(101))) static void devmem_safe_load(void)
{
    const char *track_setting = getenv("DEVMEM_SAFE_TRACK");
    track_all_nvidia_nodes = track_setting != NULL && strcmp(track_setting, "all") == 0;
    glibc_memcpy = (void *(*)(void *, const void *, size_t))dlsym(RTLD_NEXT, "memcpy");
    glibc_memmove = (void *(*)(void *, const void *, size_t))dlsym(RTLD_NEXT, "memmove");
    glibc_memset = (void *(*)(void *, int, size_t))dlsym(RTLD_NEXT, "memset");
    glibc_mmap = (void *(*)(void *, size_t, int, int, int, off_t))dlsym(RTLD_NEXT, "mmap");
    glibc_munmap = (int (*)(void *, size_t))dlsym(RTLD_NEXT, "munmap");
    char message[200];
    int length = snprintf(message, sizeof message,
                          "[devmem_safe] loaded pid=%d track=%s glibc_resolved=%d\n", (int)getpid(),
                          track_all_nvidia_nodes ? "all_nvidia_nodes" : "nvidia_gpu_nodes",
                          glibc_memcpy && glibc_memmove && glibc_memset && glibc_mmap && glibc_munmap);
    if (length > 0)
        write(2, message, (size_t)length);
}

__attribute__((destructor)) static void devmem_safe_exit(void)
{
    char message[300];
    int length = snprintf(message, sizeof message,
                          "[devmem_safe] exit pid=%d recorded_mappings=%lu aligned_memcpy=%lu "
                          "aligned_memmove=%lu aligned_memset=%lu aligned_bytes=%lu\n",
                          (int)getpid(), recorded_mapping_total, aligned_memcpy_calls,
                          aligned_memmove_calls, aligned_memset_calls, aligned_bytes);
    if (length > 0)
        write(2, message, (size_t)length);
}
