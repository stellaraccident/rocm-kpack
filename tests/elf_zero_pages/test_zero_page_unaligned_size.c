/*
 * Test case: Unaligned size (like .hip_fatbin)
 *
 * Section .testdata:
 * - Starts at 0x10000 (page-aligned)
 * - Size: 0x2310 (NOT page-aligned, 784 bytes into last page)
 *
 * Conservative zero-page should:
 * - Zero-page [0x0, 0x2000) (full pages = 8192 bytes)
 * - Keep bytes [0x2000, 0x2310) in file (last partial page = 784 bytes)
 *
 * Expected after conservative zero-page:
 * - Bytes [0, 0x2000): zeros (8192 bytes)
 * - Bytes [0x2000, 0x2310): original pattern (784 bytes)
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>

/* Page-aligned start, but odd size */
__attribute__((section(".testdata"), aligned(4096)))
const uint8_t test_data[0x2310] = {
    /* First 16 bytes: magic pattern */
    0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE,
    0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0,
    /* Rest: 0x42 */
    [16 ... 0x230F] = 0x42
};

int main(void) {
    printf("=== Zero-Page Test: Aligned Start, Unaligned Size ===\n");
    printf("Section: .testdata at %p\n", test_data);
    printf("Size: 0x%zx (%zu bytes)\n", sizeof(test_data), sizeof(test_data));
    printf("Aligned pages: [0x0, 0x2000) = 8192 bytes\n");
    printf("Partial page:  [0x2000, 0x2310) = 784 bytes\n\n");

    /* Calculate checksums for different regions */
    uint32_t sum_aligned = 0;
    uint32_t sum_partial = 0;

    /* First 0x2000 bytes (full pages) */
    for (size_t i = 0; i < 0x2000; i++) {
        sum_aligned += test_data[i];
    }

    /* Last 0x310 bytes (partial page) */
    for (size_t i = 0x2000; i < sizeof(test_data); i++) {
        sum_partial += test_data[i];
    }

    /* Check specific bytes */
    printf("First byte (aligned):      0x%02x\n", test_data[0]);
    printf("Last byte of aligned:      0x%02x\n", test_data[0x1fff]);
    printf("First byte of partial:     0x%02x\n", test_data[0x2000]);
    printf("Last byte (partial):       0x%02x\n", test_data[sizeof(test_data)-1]);
    printf("\nChecksum [0, 0x2000):      0x%08x (should be 0 if zero-paged)\n", sum_aligned);
    printf("Checksum [0x2000, 0x2310): 0x%08x (should be non-zero, preserved)\n\n", sum_partial);

    /* Expected partial page checksum: 784 * 0x42 = 0x8230 */
    const uint32_t expected_partial = 784 * 0x42;

    /* Verify result */
    if (sum_aligned == 0 && sum_partial == expected_partial) {
        printf("✓ SUCCESS: Conservative zero-paging worked correctly\n");
        printf("  - Full pages zero-paged\n");
        printf("  - Partial page preserved\n");
        return 0;
    } else if (sum_aligned > 0) {
        printf("✗ FAIL: Aligned pages not zero-paged (checksum = 0x%08x)\n", sum_aligned);
        return 1;
    } else if (sum_partial != expected_partial) {
        printf("✗ FAIL: Partial page corrupted (expected 0x%08x, got 0x%08x)\n",
               expected_partial, sum_partial);
        return 1;
    } else {
        printf("✗ FAIL: Unexpected state\n");
        return 1;
    }
}