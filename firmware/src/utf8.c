// ==============================================
// utf8.c — UTF-8 Multi-byte Sequence Handling
// ==============================================
#include "utf8.h"
#include "config.h"

// Scans 2-byte UTF-8 sequences and returns a TranslateResult
// containing the closest ASCII equivalent. Priorities:
// 1. Direct hardware mappings based on SYSTEM_LOCALE
// 2. Overstrike approximations
// 3. Simple ASCII flattening
TranslateResult flattenUtf8(const uint8_t *buf, uint8_t len) {

    // Reject any more then 2 Byte
    if (len != 2) return result0();

    // ==============================================
    // Locale-Specific Hardware Mappings
    // ==============================================
    
    if (SYSTEM_LOCALE == LOCALE_GERMAN) {
        if (buf[0] == 0xC3) {
            switch (buf[1]) {
                case 0xA4: return result1(0x27); // ä -> '{' position
                case 0xB6: return result1(0x3B); // ö -> '|' position
                case 0xBC: return result1(0x7C); // ü -> '}' position
                case 0x84: return result1(0x22); // Ä -> '[' position
                case 0x96: return result1(0x3A); // Ö -> '\' position
                case 0x9C: return result1(0x7B); // Ü -> ']' position
                case 0x9F: return result1(0x2D); // ß -> '~' position
                default: break; // Character not found in hardware, fall through to fallback
            }
        }
    }
    // Future locales (e.g., LOCALE_FRENCH) can be added as else-if blocks here

    // ==============================================
    // Universal Overstrike & Flattening Fallback
    // ==============================================

    // Block 0xC3: Common Western European (Latin-1 Supplement)
    if (buf[0] == 0xC3) {
        switch (buf[1]) {

            // --- German Umlauts (Direct Bus Overstrike) ---

            case 0xBC: return result3(0x75, 0x03, 0x40); // ü (u + BS + ")
            case 0xB6: return result3(0x6F, 0x03, 0x40); // ö (o + BS + ")
            case 0xA4: return result3(0x61, 0x03, 0x40); // ä (a + BS + ")
            case 0x9C: return result3(0x55, 0x03, 0x40); // Ü (U + BS + ")
            case 0x96: return result3(0x4F, 0x03, 0x40); // Ö (O + BS + ")
            case 0x84: return result3(0x41, 0x03, 0x40); // Ä (A + BS + ")

            case 0x9F: return result3(0x42, 0x03, 0x70); // ß -> Bp
            
            // --- Universal Letter Overstrikes ---

            // Æ (AE): 'A' (0x41) over 'E' (0x45)
            case 0x86: return result3(0x41, 0x03, 0x45); 
            
            // æ (ae): 'a' (0x61) over 'e' (0x65)
            case 0xA6: return result3(0x61, 0x03, 0x65); 
            
            // Ø (O stroke): 'O' (0x4F) over '/' (0x26)
            case 0x98: return result3(0x4F, 0x03, 0x26); 
            
            // ø (o stroke): 'o' (0x6F) over '/' (0x26)
            case 0xB8: return result3(0x6F, 0x03, 0x26); 
            
            // Ð (Eth uppercase): 'D' (0x44) over '-' (0x2F)
            case 0x90: return result3(0x44, 0x03, 0x2F); 
            
            // ÷ (Division): '-' (0x2F) over ':' (0x3A)
            case 0xB7: return result3(0x2F, 0x03, 0x3A); 

            // --- Flattening ---
            
            // × (Multiplication): Map to 'x' (0x78)
            case 0x97: return result1(0x78); 
            
            // å / Å (A ring): Flatten to a/A (no good invariant circle to overstrike)
            case 0xA5: return result1(0x61); // å -> a
            case 0x85: return result1(0x41); // Å -> A
            
            // þ / Þ (Thorn): Flatten to p/P
            case 0xBE: return result1(0x70); // þ -> p
            case 0x9E: return result1(0x50); // Þ -> P
            
            // ð (eth lowercase): Flatten to d (0x64)
            case 0xB0: return result1(0x64);
            
            // --- Standard Transliterations (Single Petal) ---
            case 0xA9: return result1(0x65);           // é -> e petal
            case 0xA8: return result1(0x65);           // è -> e petal
            case 0xB1: return result1(0x6E);           // ñ -> n petal
            case 0xA7: return result1(0x63);           // ç -> c petal

            default:
                return result0();
        }
    }

    // Block 0xC2: Symbols
    if (buf[0] == 0xC2) {
        switch (buf[1]) {
            // Degree symbol approximated by '*' petal (logic check: petals vary)
            case 0xB0: return result1(0x5B);           // ° -> * (KB1 pos)
            
            // ¢ (Cent): 'c' (0x63) over '/' (0x26)
            case 0xA2: return result3(0x63, 0x03, 0x26); 
            
            // £ (Pound): 'L' (0x4C) over '-' (0x2F)
            case 0xA3: return result3(0x4C, 0x03, 0x2F); 
            
            // ¥ (Yen): 'Y' (0x59) over '=' (0x29) 
            case 0xA5: return result3(0x59, 0x03, 0x29); 
            
            // © (Copyright): 'c' (0x63) over 'O' (0x4F)
            case 0xA9: return result3(0x63, 0x03, 0x4F); 
            
            // ® (Registered): 'R' (0x52) over 'O' (0x4F)
            case 0xAE: return result3(0x52, 0x03, 0x4F); 
            
            // ± (Plus-Minus): '+' (0x5D) over '_' (0x3F)
            case 0xB1: return result3(0x5D, 0x03, 0x3F); 
            
            // ¶ (Pilcrow/Paragraph): 'P' (0x50) over 'I' (0x49)
            case 0xB6: return result3(0x50, 0x03, 0x49);

            // ¡ (Inverted !): Map to standard '!' (0x21)
            case 0xA1: return result1(0x21); 
            
            // « (Left Guillemet): Map to '"' (0x40)
            case 0xAB: return result1(0x40); 
            
            // » (Right Guillemet): Map to '"' (0x40)
            case 0xBB: return result1(0x40); 
            
            // ¿ (Inverted ?): Map to standard '?' (0x5F)
            case 0xBF: return result1(0x5F); 
            
            // µ (Micro): Map to 'u' (0x75)
            case 0xB5: return result1(0x75);
            
            default:
                return result0();
        }
    }

    return result0();
}