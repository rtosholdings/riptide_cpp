#pragma once

// Provides the ability to add tests for a collection of types.

#define ADD_TYPED_TESTS_CAT__(X, Y) X##Y
#define ADD_TYPED_TESTS_CAT_(X, Y) ADD_TYPED_TESTS_CAT__(X, Y)
#define ADD_TYPED_TESTS_NS_(FN_NAME) ADD_TYPED_TESTS_CAT_(add_typed_tests__, FN_NAME)

#define DECL_ADD_TYPED_TESTS(FN_NAME) \
    namespace details::ADD_TYPED_TESTS_NS_(FN_NAME) \
    { \
        namespace details \
        { \
            template <typename Tst, typename Type> \
            void add_typed_test(Tst & tst) \
            { \
                tst = FN_NAME<Type>; \
            } \
            template <typename Tst> \
            void add_typed_tests(Tst &) \
            { \
            } \
            template <typename Tst, typename Type, typename... Types> \
            void add_typed_tests(Tst & tst) \
            { \
                add_typed_test<Tst, Type>(tst); \
                add_typed_tests<Tst, Types...>(tst); \
            } \
        } \
        template <typename Tst, typename... Types> \
        void add_typed_tests(Tst && tst, std::tuple<Types...> const & types) \
        { \
            details::add_typed_tests<Tst, Types...>(tst); \
        } \
    }

#define ADD_TYPED_TESTS(FN_NAME) details::ADD_TYPED_TESTS_NS_(FN_NAME)::add_typed_tests