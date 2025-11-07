/*
 * Test program for elf_modify_load section mapping.
 *
 * This program has a structure similar to __CudaFatBinaryWrapper that contains
 * a pointer. The pointer will be updated via relocation to point to a custom
 * section that we map to a new PT_LOAD segment.
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>

// Structure similar to __CudaFatBinaryWrapper
struct TestWrapper {
    uint32_t magic;
    uint32_t version;
    const char* data_ptr;  // This will point to mapped section
    void* reserved;
};

// Dummy data in .data section to generate a relocation
static const char dummy_data[8] = "dummy";

// Initialize the wrapper in its own section
// The data_ptr will be relocated to point to .custom_data section
//
// IMPORTANT: For PIE binaries, pointers MUST be initialized with relocatable addresses
// (like &dummy_data) to generate R_X86_64_RELATIVE relocations. The elf_modify_load
// set-pointer tool requires existing relocations for PIE/shared libraries and will fail
// with a clear error if the pointer location has no relocation entry.
//
// Constant addresses (like 0x1000) do NOT generate relocations and will cause the tool
// to error out, preventing creation of binaries that would crash at runtime.
struct TestWrapper test_wrapper __attribute__((section(".test_wrapper"))) = {
    .magic = 0x54455354,  // "TEST" in little-endian
    .version = 1,
    .data_ptr = dummy_data,  // Points to dummy_data, generates relocation in PIE
    .reserved = NULL
};

int main() {
    printf("Test Mapped Section\n");
    printf("===================\n\n");

    printf("Wrapper structure:\n");
    printf("  Address: %p\n", (void*)&test_wrapper);
    printf("  Magic: 0x%08x\n", test_wrapper.magic);
    printf("  Version: %u\n", test_wrapper.version);
    printf("  Data pointer: %p\n", (void*)test_wrapper.data_ptr);
    printf("  Reserved: %p\n", test_wrapper.reserved);
    printf("\n");

    // Verify magic
    if (test_wrapper.magic != 0x54455354) {
        fprintf(stderr, "ERROR: Invalid magic number\n");
        return 1;
    }

    // Check if pointer is NULL (would indicate relocation not updated)
    if (test_wrapper.data_ptr == NULL || test_wrapper.data_ptr == (const char*)0x1000) {
        fprintf(stderr, "ERROR: Data pointer not relocated (ptr=%p)\n",
                (void*)test_wrapper.data_ptr);
        return 1;
    }

    printf("Reading data from mapped section:\n");
    printf("  Pointer value: %p\n", (void*)test_wrapper.data_ptr);

    // Try to read the string
    // The .custom_data section will contain "Hello from mapped section!"
    const char* expected = "Hello from mapped section!";
    size_t expected_len = strlen(expected);

    // Read first few bytes to verify
    printf("  Data content: \"");
    for (size_t i = 0; i < expected_len && test_wrapper.data_ptr[i] != '\0'; i++) {
        putchar(test_wrapper.data_ptr[i]);
    }
    printf("\"\n\n");

    // Verify content
    if (strncmp(test_wrapper.data_ptr, expected, expected_len) != 0) {
        fprintf(stderr, "ERROR: Data mismatch!\n");
        fprintf(stderr, "  Expected: \"%s\"\n", expected);
        fprintf(stderr, "  Got: \"%.30s\"\n", test_wrapper.data_ptr);
        return 1;
    }

    printf("✅ SUCCESS: Mapped section data verified!\n");
    return 0;
}
