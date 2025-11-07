/*
 * Test case: Both start and size unaligned (worst case)
 *
 * Section .testdata:
 * - Starts at 0x10080 (NOT page-aligned)
 * - Size: 0x2310 (NOT page-aligned)
 *
 * Conservative zero-page should:
 * - Keep [0x0, 0x1000) in file (prefix before first page boundary)
 * - Zero-page [0x1000, 0x2000) (full pages)
 * - Keep [0x2000, 0x2310) in file (partial page at end)
 *
 * Expected after conservative zero-page:
 * - Bytes [0, 0xf80): original (3968 bytes)
 * - Bytes [0xf80, 0x1f80): zeros (4096 bytes = 1 full page)
 * - Bytes [0x1f80, 0x2310): original (912 bytes)
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>

/* Start with padding to make it unaligned */
__attribute__((section(".testdata"), aligned(4096)))
const uint8_t padding[128] = { [0 ... 127] = 0xFF };

__attribute__((section(".testdata")))
const uint8_t test_data[0x2310 - 128] = {
    /* First 16 bytes: magic pattern */
    0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE,
    0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0,
    /* Rest: 0x42 */
    [16 ... 0x2310 - 128 - 1] = 0x42
};

int main(void) {
    printf("=== Zero-Page Test: Unaligned Start AND Unaligned Size ===\n");
    printf("Padding at: %p (%zu bytes)\n", padding, sizeof(padding));
    printf("Data at:    %p (%zu bytes)\n", test_data, sizeof(test_data));
    printf("Total size: %zu bytes (0x%zx)\n\n",
           sizeof(padding) + sizeof(test_data),
           sizeof(padding) + sizeof(test_data));

    printf("Expected layout:\n");
    printf("  [0x0, 0xf80):      original (prefix)\n");
    printf("  [0xf80, 0x1f80):   zeros (1 full page)\n");
    printf("  [0x1f80, 0x2310):  original (suffix)\n\n");

    /* Calculate checksums for three regions */
    uint32_t sum_prefix = 0;    /* [0, 0xf80) */
    uint32_t sum_aligned = 0;   /* [0xf80, 0x1f80) */
    uint32_t sum_suffix = 0;    /* [0x1f80, 0x2310) */

    size_t total_size = sizeof(padding) + sizeof(test_data);

    for (size_t i = 0; i < total_size; i++) {
        uint8_t byte;
        if (i < sizeof(padding)) {
            byte = padding[i];
        } else {
            byte = test_data[i - sizeof(padding)];
        }

        if (i < 0xf80) {
            sum_prefix += byte;
        } else if (i < 0x1f80) {
            sum_aligned += byte;
        } else {
            sum_suffix += byte;
        }
    }

    printf("Checksum [0, 0xf80):       0x%08x (should be non-zero)\n", sum_prefix);
    printf("Checksum [0xf80, 0x1f80):  0x%08x (should be 0 if zero-paged)\n", sum_aligned);
    printf("Checksum [0x1f80, 0x2310): 0x%08x (should be non-zero)\n\n", sum_suffix);

    /* Verify result */
    int prefix_ok = (sum_prefix > 0);
    int aligned_ok = (sum_aligned == 0);
    int suffix_ok = (sum_suffix > 0);

    if (prefix_ok && aligned_ok && suffix_ok) {
        printf("✓ SUCCESS: Conservative zero-paging worked correctly\n");
        printf("  - Prefix preserved\n");
        printf("  - Aligned middle zero-paged\n");
        printf("  - Suffix preserved\n");
        return 0;
    } else if (!aligned_ok) {
        printf("✗ FAIL: Aligned region not zero-paged\n");
        return 1;
    } else if (!prefix_ok || !suffix_ok) {
        printf("✗ FAIL: Unaligned regions corrupted\n");
        return 1;
    } else {
        printf("✗ FAIL: Unexpected state\n");
        return 1;
    }
}