#pragma once
ANY_TWO_FUNC GetComparisonOpFast(int func, int scalarMode, int numpyInType1, int numpyInType2, int numpyOutType,
                                 int * wantedOutType);
ANY_TWO_FUNC GetComparisonOpSlow(int func, int scalarMode, int numpyInType1, int numpyInType2, int numpyOutType,
                                 int * wantedOutType);
