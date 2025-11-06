/*
 * Test case: Page-aligned section (baseline case)
 *
 * Section .testdata:
 * - Starts at 0x10000 (page-aligned)
 * - Size: 0x1000 (page-aligned)
 *
 * Expected after zero-page:
 * - All 4096 bytes should be zeros
 * - Checksum: 0x0
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>

/* Page-aligned section */
__attribute__((section(".testdata"), aligned(4096)))
const uint8_t test_data[4096] = {
    /* First 16 bytes: magic pattern */
    0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE,
    0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0,
    /* Rest: 0x42 */
    [16 ... 4095] = 0x42
};

int main(void) {
    printf("=== Zero-Page Test: Aligned Start, Aligned Size ===\n");
    printf("Section: .testdata at %p\n", test_data);
    printf("Expected range: [0x0, 0x1000) = 4096 bytes\n\n");

    /* Calculate checksum */
    uint32_t sum = 0;
    for (int i = 0; i < 4096; i++) {
        sum += test_data[i];
    }

    /* Check first and last bytes */
    printf("First byte: 0x%02x\n", test_data[0]);
    printf("Last byte:  0x%02x\n", test_data[4095]);
    printf("Checksum:   0x%08x\n\n", sum);

    /* Verify result */
    if (sum == 0) {
        printf("✓ SUCCESS: All bytes zero-paged\n");
        return 0;
    } else if (sum == 0x42690) {
        printf("✗ FAIL: No zero-paging applied (original data present)\n");
        return 1;
    } else {
        printf("✗ FAIL: Unexpected checksum (partial zero-paging?)\n");
        return 1;
    }
}