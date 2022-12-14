#include <math.h>
#include "CommonInc.h"
#include "missing_values.h"
namespace Heap
{
    static constexpr size_t heapNullIndex = static_cast<size_t>(-1);

    template <typename Element, typename CompareElements>
    struct Heap
    {
        explicit Heap(size_t max_size, Element ** elements)
            : max_size{ max_size }
            , _elements{ elements }
            , compare{ CompareElements() } {};

        size_t max_size;
        Element ** _elements;
        CompareElements compare;
        size_t _current_size = static_cast<size_t>(0);

        size_t size() const
        {
            return _current_size;
        }

        bool full() const
        {
            return (size() >= max_size);
        }

        bool empty() const
        {
            return (size() == 0);
        }

        size_t _increase_size()
        {
            if (full())
            {
                // std::cout << "Heap is full already!" << std::endl;
                return static_cast<size_t>(-1);
            }
            return ++_current_size;
        }

        size_t _decrease_size()
        {
            if (empty())
            {
                // std::cout << "Heap is empty already!" << std::endl;
                return static_cast<size_t>(-1);
            }
            return --_current_size;
        }

        size_t Push(Element * const value)
        {
            if (full())
            {
                // std::cout << "Heap is full already!" << std::endl;
                return static_cast<size_t>(-1);
            }

            _elements[size()] = value;
            NotifyIndexChange(value, size());
            SiftUp(size());
            return _increase_size();
        }

        size_t PopBack()
        {
            if (empty())
            {
                // std::cout << "Heap is empty already!" << std::endl;
                return static_cast<size_t>(-1);
            }
            NotifyIndexChange(_elements[size() - 1], heapNullIndex);
            _elements[size() - 1] = nullptr;
            return _decrease_size();
        }

        void Erase(size_t index)
        {
            if (index == heapNullIndex)
            {
                // std::cout << "Trying to erase something not in the heap\n";
                return;
            }
            // assert(index < size());
            Swap_elements(index, size() - 1);
            PopBack();
            // only heapify if this wasn't the last element in array
            if (index < size())
            {
                SiftDown(index);
                SiftUp(index);
            }
        }

        void Clean()
        {
            while (size())
            {
                PopBack();
            }
        }

        void ApplyElementChange(size_t index)
        {
            if (index == heapNullIndex)
            {
                // std::cout << "Trying to modify something not in the heap\n";
                return;
            }
            // assert(index < size());
            SiftDown(index);
            SiftUp(index);
        }

        Element * PopTop()
        {
            if (size())
            {
                Element * to_return = top();
                Erase(0);
                return to_return;
            }
            else
            {
                return nullptr;
            }
        }

        Element * top() const
        {
            if (size())
            {
                return _elements[0];
            }
            else
            {
                return nullptr;
            }
        }

        size_t SiftUp(size_t index)
        {
            while (index && Compare_elements(parent(index), index))
            {
                Swap_elements(parent(index), index);
                index = parent(index);
            }
            return index;
        }

        void SiftDown(size_t index)
        {
            while (right_child(index) <= size())
            {
                size_t index_to_swap = left_child(index);
                if (right_child(index) < size() && Compare_elements(left_child(index), right_child(index)))
                {
                    index_to_swap = right_child(index);
                }
                if (Compare_elements(index_to_swap, index))
                {
                    break;
                }
                Swap_elements(index, index_to_swap);
                index = index_to_swap;
            }
        }

        size_t parent(size_t index) const
        {
            return (index - 1) / 2;
        }

        size_t left_child(size_t index) const
        {
            return 2 * index + 1;
        }
        size_t right_child(size_t index) const
        {
            return 2 * index + 2;
        }

        bool Compare_elements(size_t first_index, size_t second_index) const
        {
            return compare(_elements[first_index], _elements[second_index]);
        }

        void NotifyIndexChange(Element * const element, size_t new_element_index) const
        {
            element->ChangePositionInHeap(new_element_index);
        }

        void Swap_elements(size_t first_index, size_t second_index)
        {
            Element * const buff_element = _elements[first_index];
            _elements[first_index] = _elements[second_index];
            _elements[second_index] = buff_element;
            NotifyIndexChange(_elements[first_index], first_index);
            NotifyIndexChange(_elements[second_index], second_index);
        }
    };

    template <typename T>
    struct HeapElement
    {
        T const * data;
        size_t heap_position;

        HeapElement(T const * data)
            : data{ data }
            , heap_position{ heapNullIndex } {};

        void Reinstantiate(T const * const data)
        {
            this->data = data;
        }

        void ChangePositionInHeap(size_t new_index)
        {
            this->heap_position = new_index;
            return;
        }
    };

    template <typename Element, typename Compare>
    struct CompareElements
    {
        Compare compare;

        CompareElements()
            : compare{ Compare() } {};

        bool operator()(const Element * first, const Element * second) const
        {
            return compare(*(first->data), *(second->data));
        }
    };
}

namespace RollingQuantile
{
    enum class StructureLocation
    {
        InMinHeap,
        InMaxHeap,
        InNanList,
        NotInStructure
    };

    template <typename T>
    struct StructureElement : public Heap::HeapElement<T>
    {
        StructureLocation structure_location;

        // stores the order in which elements were added to the structure
        StructureElement * next_in_structure = nullptr;

        StructureElement * next_in_list;
        StructureElement * prev_in_list;

        StructureElement(T const * const data)
            : Heap::HeapElement<T>{ data }
            , structure_location{ StructureLocation::NotInStructure } {};
    };

    template <typename Element>
    struct List
    {
        Element * const head;
        Element * const tail;
        size_t _curr_size;

        List()
            : head{ new Element(nullptr) }
            , tail{ new Element(nullptr) }
            , _curr_size{ 0 }
        {
            InitHeadTail();
        };

        void InitHeadTail()
        {
            head->prev_in_list = nullptr;
            head->next_in_list = tail;

            tail->prev_in_list = head;
            tail->next_in_list = nullptr;
        }

        ~List()
        {
            delete head;
            delete tail;
        }

        void Clean()
        {
            Element * curr_element = head;
            while (curr_element != tail)
            {
                Element * next_element = curr_element->next_in_list;
                curr_element->next_in_list = nullptr;
                curr_element->prev_in_list = nullptr;
                curr_element->structure_location = StructureLocation::NotInStructure;
                curr_element = next_element;
            }
            _curr_size = 0;
            InitHeadTail();
        }

        void Push(Element * new_element)
        {
            Element * old_last_element = this->tail->prev_in_list;

            old_last_element->next_in_list = new_element;
            this->tail->prev_in_list = new_element;

            new_element->next_in_list = this->tail;
            new_element->prev_in_list = old_last_element;

            new_element->structure_location = StructureLocation::InNanList;
            _curr_size++;
        }

        void Erase(Element * element_to_remove)
        {
            // assert(element_to_remove->next_in_list);
            // assert(element_to_remove->prev_in_list);
            // assert(_curr_size);

            Element * next_element = element_to_remove->next_in_list;
            Element * prev_element = element_to_remove->prev_in_list;

            prev_element->next_in_list = next_element;
            next_element->prev_in_list = prev_element;

            element_to_remove->next_in_list = nullptr;
            element_to_remove->prev_in_list = nullptr;

            element_to_remove->structure_location = StructureLocation::NotInStructure;

            _curr_size--;
        }

        size_t size() const
        {
            return _curr_size;
        }
    };

    static constexpr long double precision_epsilon = static_cast<long double>(1e-3);

    inline long double FractionalIndex(size_t n_elements, long double quantile)
    {
        return (n_elements - 1) * quantile;
    }

    inline size_t IntegralIndex(size_t n_elements, long double quantile)
    {
        long double int_part;
        long double frac_part = modfl(FractionalIndex(n_elements, quantile), &int_part);

        if (frac_part > (1 - precision_epsilon))
        {
            return (size_t)(int_part + 1);
        }
        else
        {
            return (size_t)(int_part);
        }
    }

    template <typename T, typename U>
    struct RollingQuantile
    {
        using RollingElement = StructureElement<T>;
        using MaxHeap = Heap::Heap<RollingElement, Heap::CompareElements<RollingElement, std::less<T>>>;
        using MinHeap = Heap::Heap<RollingElement, Heap::CompareElements<RollingElement, std::greater<T>>>;

        using SplitFunction = U (*)(T, T);

        size_t window_size;
        long double quantile;
        RollingElement * all_elements;
        MaxHeap max_heap;
        MinHeap min_heap;
        size_t min_count;
        SplitFunction quantile_split_function;

        // Used only for comparison between heads of heaps
        Heap::CompareElements<RollingElement, std::less<T>> element_compare;
        List<RollingElement> nan_list;

        RollingElement * oldest_element = nullptr;
        RollingElement * newest_element = nullptr;

        RollingQuantile(size_t window_size, long double quantile, RollingElement * all_elements,
                        RollingElement ** all_elements_pointers, size_t max_heap_size, size_t min_heap_size, size_t min_count,
                        SplitFunction quantile_split_function)
            : window_size{ window_size }
            , quantile{ quantile }
            , all_elements{ all_elements }
            , max_heap{ max_heap_size, all_elements_pointers }
            , min_heap{ min_heap_size, all_elements_pointers + max_heap_size }
            , min_count{ min_count }
            , quantile_split_function{ quantile_split_function }
            , element_compare{ Heap::CompareElements<RollingElement, std::less<T>>() }
            , nan_list{ List<RollingElement>() } {
                // assert(window_size > 1);
                // assert(min_count);
            };

        void Clean()
        {
            max_heap.Clean();
            min_heap.Clean();
            nan_list.Clean();
            oldest_element = nullptr;
            newest_element = nullptr;
        }

        inline long double FractionalQuantileIndex(size_t n_elements) const
        {
            return FractionalIndex(n_elements, this->quantile);
        }

        size_t QuantileIndex(size_t n_elements) const
        {
            return IntegralIndex(n_elements, this->quantile);
        }

        bool QuantileIndexIsIntegral(size_t n_elements) const
        {
            long double int_part;
            long double frac_part = modfl(FractionalQuantileIndex(n_elements), &int_part);

            return ((frac_part < precision_epsilon) || (frac_part > (1 - precision_epsilon)));
        }

        U GetQuantile() const
        {
            size_t n_non_nan_elements = total_non_nan_elements();
            if (n_non_nan_elements < min_count)
            {
                return GET_INVALID(U{});
            }

            if (QuantileIndexIsIntegral(n_non_nan_elements))
            {
                // assert(not max_heap.empty());
                return (U)(*max_heap.top()->data);
            }
            else
            {
                // assert(not max_heap.empty());
                // assert(not min_heap.empty());

                return quantile_split_function(*max_heap.top()->data, *min_heap.top()->data);
            }
        }

        void PossiblySwapHeapHeads()
        {
            if (min_heap.size() && max_heap.size() && element_compare(min_heap.top(), max_heap.top()))
            {
                std::swap(min_heap._elements[0], max_heap._elements[0]);
                min_heap.top()->structure_location = StructureLocation::InMinHeap;
                max_heap.top()->structure_location = StructureLocation::InMaxHeap;

                min_heap.ApplyElementChange(0);
                max_heap.ApplyElementChange(0);
            }
        }

        void ApplyChangeInCorrespondingHeap(RollingElement * updated_element)
        {
            if (updated_element->structure_location == StructureLocation::InMinHeap)
            {
                min_heap.ApplyElementChange(updated_element->heap_position);
            }
            else if (updated_element->structure_location == StructureLocation::InMaxHeap)
            {
                max_heap.ApplyElementChange(updated_element->heap_position);
            }
            else
            {
                // assert(0);
            }
        }

        size_t total_non_nan_elements() const
        {
            return min_heap.size() + max_heap.size();
        }

        size_t n_nan_elements() const
        {
            return nan_list.size();
        }

        size_t total_elements() const
        {
            return total_non_nan_elements() + n_nan_elements();
        }

        U Update(T const * const new_data)
        {
            if (total_elements() < window_size)
            {
                return NanUpdateNotFull(new_data);
            }
            else
            {
                return NanUpdateFull(new_data);
            }
        }

        U NanUpdateNotFull(T const * const new_data)
        {
            RollingElement * new_element = &all_elements[total_elements()];
            *new_element = RollingElement(new_data);

            if (not total_elements())
            {
                this->oldest_element = new_element;
            }
            else
            {
                this->newest_element->next_in_structure = new_element;
            }
            this->newest_element = new_element;

            if (not riptide::invalid_for_type<T>::is_valid(*new_data))
            {
                // Put new element in NaN array, don't update heaps.
                PushToNanList(new_element);
            }
            else
            {
                PushToCorrectHeap(new_element);
            }

            return GetQuantile();
        }

        U NanUpdateFull(T const * const new_data)
        {
            // assert(window_size == total_elements());

            RollingElement * updated_element = this->oldest_element;
            bool old_element_is_nan = (updated_element->structure_location == StructureLocation::InNanList);

            updated_element->Reinstantiate(new_data);

            bool new_element_is_nan = (not riptide::invalid_for_type<T>::is_valid(*new_data));

            this->oldest_element = updated_element->next_in_structure;
            this->newest_element->next_in_structure = updated_element;
            this->newest_element = updated_element;

            if ((not old_element_is_nan) && (not new_element_is_nan))
            {
                ApplyChangeInCorrespondingHeap(updated_element);
                PossiblySwapHeapHeads();
            }
            else if ((old_element_is_nan) && (not new_element_is_nan))
            {
                EraseFromNanList(updated_element);
                PushToCorrectHeap(updated_element);
            }
            else if ((not old_element_is_nan) && (new_element_is_nan))
            {
                EraseFromCorrespondingHeap(updated_element);
                PushToNanList(updated_element);
                RebalanceHeaps();
            }
            // else if (old_element_is_nan && new_element_is_nan)
            // {
            //     // do nothing
            // }

            return GetQuantile();
        }

        void RebalanceHeaps()
        {
            if (not total_non_nan_elements())
            {
                return;
            }
            size_t quantile_index = QuantileIndex(total_non_nan_elements());
            if (max_heap.size() > quantile_index + 1)
            {
                //  move from max_heap to min_heap
                RollingElement * element_to_move = max_heap.top();
                EraseFromCorrespondingHeap(element_to_move);
                PushToMinHeap(element_to_move);
            }
            else if (max_heap.size() < quantile_index + 1)
            {
                //  move from min_heap to max_heap
                RollingElement * element_to_move = min_heap.top();
                EraseFromCorrespondingHeap(element_to_move);
                PushToMaxHeap(element_to_move);
            }

            // assert(max_heap.size() == quantile_index + 1);
            return;
        }

        void EraseFromCorrespondingHeap(RollingElement * updated_element)
        {
            if (updated_element->structure_location == StructureLocation::InMinHeap)
            {
                min_heap.Erase(updated_element->heap_position);
            }
            else if (updated_element->structure_location == StructureLocation::InMaxHeap)
            {
                max_heap.Erase(updated_element->heap_position);
            }
            else
            {
                // assert(0);
            }
            updated_element->structure_location = StructureLocation::NotInStructure;
        }

        void EraseFromNanList(RollingElement * element_to_remove)
        {
            // assert(element_to_remove->structure_location == StructureLocation::InNanList);
            // assert(n_nan_elements());
            nan_list.Erase(element_to_remove);
        }

        void PushToCorrectHeap(RollingElement * element_to_push)
        {
            // +1 because adding new element
            size_t quantile_index = QuantileIndex(total_non_nan_elements() + 1);

            if (max_heap.size() > quantile_index)
            {
                PushToMinHeap(element_to_push);
            }
            else
            {
                PushToMaxHeap(element_to_push);
            }

            PossiblySwapHeapHeads();
        }

        void PushToMaxHeap(RollingElement * element)
        {
            max_heap.Push(element);
            element->structure_location = StructureLocation::InMaxHeap;
        }

        void PushToMinHeap(RollingElement * element)
        {
            min_heap.Push(element);
            element->structure_location = StructureLocation::InMinHeap;
        }

        void PushToNanList(RollingElement * element)
        {
            nan_list.Push(element);
        }
    };
}
