#ifndef CUSTOM_ITER_H_
#define CUSTOM_ITER_H_

#include <iterator>

class custom_iter
{
private:
	int value_;
	class intholder
	{
		int value_;
	public:
		__device__ intholder(int value) : value_(value) {}
		__device__ int operator*() { return value_; }
	};
public:
	// Previously provided by std::iterator
	typedef int                     value_type;
	typedef std::ptrdiff_t          difference_type;
	typedef int*                    pointer;
	typedef int&                    reference;
	typedef std::input_iterator_tag iterator_category;

	explicit __device__ custom_iter(int value) : value_(value) {}
	__device__ int operator*() const { return value_; }
	__device__ bool operator==(const custom_iter& other) const { return value_ == other.value_; }
	__device__ bool operator!=(const custom_iter& other) const { return !(*this == other); }
	__device__ intholder operator++(int)
	{
		intholder ret(value_);
		++*this;
		return ret;
	}
	__device__ custom_iter& operator++()
	{
		++value_;
		return *this;
	}
};

#endif /* CUSTOM_ITER_H_ */