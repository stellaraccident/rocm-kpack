/*
 * Test case: Unaligned start address
 *
 * Section .testdata:
 * - Starts at 0x10080 (NOT page-aligned, 128 bytes into page)
 * - Size: 0x2000 (page-aligned size)
 *
 * Conservative zero-page should:
 * - Keep bytes [0x0, 0x1000) in file (0x80 bytes before first page boundary)
 * - Zero-page [0x1000, 0x2000) (full pages)
 *
 * Expected after conservative zero-page:
 * - Bytes [0, 0xf80): original pattern (first 0xf80 bytes)
 * - Bytes [0xf80, 0x2000): zeros (remaining 0x1080 bytes)
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>

/* Unaligned section: starts 128 bytes into a page */
__attribute__((section(".testdata"), aligned(4096)))
const uint8_t padding[128] = { [0 ... 127] = 0xFF };

__attribute__((section(".testdata")))
const uint8_t test_data[8192 - 128] = {
    /* First 16 bytes: magic pattern */
    0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE,
    0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0,
    /* Rest: 0x42 */
    [16 ... 8191 - 128] = 0x42
};

int main(void) {
    printf("=== Zero-Page Test: Unaligned Start, Aligned Size ===\n");
    printf("Padding at: %p (%zu bytes)\n", padding, sizeof(padding));
    printf("Data at:    %p (%zu bytes)\n", test_data, sizeof(test_data));
    printf("Total size: %zu bytes\n\n", sizeof(padding) + sizeof(test_data));

    /* Check padding (should stay as 0xFF) */
    int padding_ok = 1;
    for (size_t i = 0; i < sizeof(padding); i++) {
        if (padding[i] != 0xFF) {
            padding_ok = 0;
            break;
        }
    }

    /* Calculate checksums for different regions */
    uint32_t sum_before_aligned = 0;
    uint32_t sum_aligned = 0;

    /* First 0xf80 bytes (before first page boundary) */
    for (size_t i = 0; i < 0xf80; i++) {
        if (i < sizeof(padding)) {
            sum_before_aligned += padding[i];
        } else {
            sum_before_aligned += test_data[i - sizeof(padding)];
        }
    }

    /* Remaining bytes (aligned region) */
    for (size_t i = 0xf80; i < sizeof(padding) + sizeof(test_data); i++) {
        if (i < sizeof(padding)) {
            sum_aligned += padding[i];
        } else {
            sum_aligned += test_data[i - sizeof(padding)];
        }
    }

    printf("Padding preserved: %s\n", padding_ok ? "yes" : "NO");
    printf("Checksum [0, 0xf80):       0x%08x (should be non-zero)\n", sum_before_aligned);
    printf("Checksum [0xf80, 0x2000):  0x%08x (should be 0 if zero-paged)\n\n", sum_aligned);

    /* Verify result */
    if (padding_ok && sum_before_aligned > 0 && sum_aligned == 0) {
        printf("✓ SUCCESS: Conservative zero-paging worked correctly\n");
        printf("  - Unaligned prefix preserved\n");
        printf("  - Aligned region zero-paged\n");
        return 0;
    } else if (sum_before_aligned > 0 && sum_aligned > 0) {
        printf("✗ FAIL: No zero-paging applied (original data present)\n");
        return 1;
    } else {
        printf("✗ FAIL: Unexpected state\n");
        return 1;
    }
}