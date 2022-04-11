#pragma once

// Basic cross-platform definitions.

// Export DLL section
#if defined(_WIN32) && ! defined(__GNUC__)
# ifdef BUILDING_RIPTIDE_CPP
#  define DllExport __declspec(dllexport)
# else
#  define DllExport __declspec(dllimport)
# endif
#else
# define DllExport
#endif
