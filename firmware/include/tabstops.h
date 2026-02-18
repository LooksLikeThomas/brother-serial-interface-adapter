// ==============================================
// tabstops.h — Horizontal Tab Stop Bitmask
// ==============================================
//
// Stores tab stop positions as a compact bitmask.
// Supports up to TAB_MAX_COLUMNS columns.
//
// Builder functions (tabSet, tabClear, tabInit) configure stops.
// Consumer function (tabNextStop) finds the next stop from a given column.
//
#ifndef TABSTOPS_H
#define TABSTOPS_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Widest platen at 15cpi on 12" paper — generous upper bound
#define TAB_MAX_COLUMNS 160

// ==============================================
// Tab Stop Bitmask
// ==============================================

typedef struct {
    uint8_t bits[TAB_MAX_COLUMNS / 8 + 1];
} TabStops;

// ==============================================
// Helpers
// ==============================================

// Returns true if a tab stop is set at the given column (1-based)
static inline bool tabGet(const TabStops *t, uint8_t col) {
    return t->bits[col >> 3] & (1 << (col & 7));
}

// Sets a tab stop at the given column
static inline void tabSet(TabStops *t, uint8_t col) {
    t->bits[col >> 3] |= (1 << (col & 7));
}

// Clears a tab stop at the given column
static inline void tabClear(TabStops *t, uint8_t col) {
    t->bits[col >> 3] &= ~(1 << (col & 7));
}

// Clears all tab stops
static inline void tabClearAll(TabStops *t) {
    for (uint8_t i = 0; i < sizeof(t->bits); i++) {
        t->bits[i] = 0;
    }
}

// Clears all stops then pre-fills at every `every` columns up to maxCol.
// First stop is at column (every + 1), then (2*every + 1), etc.
// If every is 0, all stops are left empty.
static inline void tabInit(TabStops *t, uint8_t every, uint8_t maxCol) {
    tabClearAll(t);
    if (every > 0) {
        for (uint8_t col = every + 1; col <= maxCol; col += every) {
            tabSet(t, col);
        }
    }
}

// Returns the next tab stop column at or after (col + 1), up to rightMargin.
// Returns 0 if no stop found.
static inline uint8_t tabNextStop(const TabStops *t, uint8_t col, uint8_t rightMargin) {
    for (uint8_t c = col + 1; c <= rightMargin; c++) {
        if (tabGet(t, c)) return c;
    }
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif // TABSTOPS_H
