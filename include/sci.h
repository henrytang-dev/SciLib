#ifndef SCI_H
#define SCI_H

#include <fstream>
#include <cmath>
#include <complex>
#include <iostream>
#include <iomanip>
#include <vector>
#include <limits>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <fcntl.h>
#include <string.h>
#include <ctype.h>

using namespace std;

// macro-like inline fn

template<typename T>
inline T SQRT(const T a) { return a * a; }

// Vector and Matrix Classes

template <class T>
class SLvector {
private:
    int nn; // size of array. upper index is nn-1
    int cap; // capacity of array
    std::unique_ptr<T[]> v;

    void reallocate(int new_capacity) {
        std::unique_ptr<T[]> new_v(new (std::nothrow) T[new_capacity]);
        if (!new_v) throw std::bad_alloc();
        if (nn > 0) {
            std::move(v.get(), v.get() + nn, new_v.get());
        }
        v.swap(new_v);
        cap = new_capacity;
    }

public:
    SLvector() noexcept;
    explicit SLvector(int n);
    SLvector(int n, const T &a);
    SLvector(int n, const T *a);
    SLvector(const SLvector &rhs);
    SLvector(SLvector &&rhs) noexcept;
    SLvector & operator=(const SLvector &rhs);
    SLvector & operator=(SLvector &&rhs) noexcept;
    SLvector(std::initializer_list<T> init);

    typedef T value_type; // make T available externally
    inline T & operator[](const int i) noexcept;
    inline const T & operator[](const int i) const noexcept;
    inline int size() const noexcept;
    inline int capacity() const noexcept;
    void reserve(int new_capacity);
    void resize(int newn);
    void assign(int newn, const T &a);
    void push_back(const T &value);
    void push_back(T &&value);
    template <class... Args>
    void emplace_back(Args&&... args);
    void shrink_to_fit();

    // Iterators
    T* begin() noexcept;
    const T* begin() const noexcept;
    T* end() noexcept;
    const T* end() const noexcept;

    ~SLvector() = default;
};

// SLvector definitions

template <class T>
SLvector<T>::SLvector() noexcept : nn(0), cap(0), v(nullptr) {}

template <class T>
SLvector<T>::SLvector(int n) : nn(n), cap(n), v(n > 0 ? std::make_unique<T[]>(n) : nullptr) {}

template <class T>
SLvector<T>::SLvector(int n, const T& a) : nn(n), cap(n), v(n > 0 ? std::make_unique<T[]>(n) : nullptr) {
    std::fill_n(v.get(), n, a);
}

template <class T>
SLvector<T>::SLvector(int n, const T *a) : nn(n), cap(n), v(n > 0 ? std::make_unique<T[]>(n) : nullptr) {
    std::copy(a, a + n, v.get());
}

template <class T>
SLvector<T>::SLvector(const SLvector<T> &rhs) : nn(rhs.nn), cap(rhs.nn), v(nn > 0 ? std::make_unique<T[]>(nn) : nullptr) {
    std::copy(rhs.v.get(), rhs.v.get() + nn, v.get());
}

template <class T>
SLvector<T>::SLvector(SLvector<T> &&rhs) noexcept : nn(rhs.nn), cap(rhs.cap), v(std::move(rhs.v)) {
    rhs.nn = 0;
    rhs.cap = 0;
}

template <class T>
SLvector<T> & SLvector<T>::operator=(const SLvector<T> &rhs) {
    if (this != &rhs) {
        if (rhs.nn > cap) {
            reallocate(rhs.nn);
        }
        nn = rhs.nn;
        std::copy(rhs.v.get(), rhs.v.get() + nn, v.get());
    }
    return *this;
}

template <class T>
SLvector<T> & SLvector<T>::operator=(SLvector<T> &&rhs) noexcept {
    if (this != &rhs) {
        nn = rhs.nn;
        cap = rhs.cap;
        v = std::move(rhs.v);
        rhs.nn = 0;
        rhs.cap = 0;
    }
    return *this;
}

template <class T>
SLvector<T>::SLvector(std::initializer_list<T> init) : nn(init.size()), cap(init.size()), v(init.size() > 0 ? std::make_unique<T[]>(init.size()) : nullptr) {
    std::copy(init.begin(), init.end(), v.get());
}

template <class T>
inline T & SLvector<T>::operator[](const int i) noexcept {
#ifdef _CHECKBOUNDS_
    if (i < 0 || i >= nn) {
        throw std::out_of_range("SLvector subscript out of bounds");
    }
#endif
    return v[i];
}

template <class T>
inline const T & SLvector<T>::operator[](const int i) const noexcept {
#ifdef _CHECKBOUNDS_
    if (i < 0 || i >= nn) {
        throw std::out_of_range("SLvector subscript out of bounds");
    }
#endif
    return v[i];
}

template <class T>
inline int SLvector<T>::size() const noexcept {
    return nn;
}

template <class T>
inline int SLvector<T>::capacity() const noexcept {
    return cap;
}

template <class T>
void SLvector<T>::reserve(int new_capacity) {
    if (new_capacity > cap) {
        reallocate(new_capacity);
    }
}

template <class T>
void SLvector<T>::resize(int newn) {
    if (newn > cap) {
        reallocate(newn);
    }
    nn = newn;
}

template <class T>
void SLvector<T>::assign(int newn, const T& a) {
    resize(newn);
    std::fill_n(v.get(), newn, a);
}

template <class T>
void SLvector<T>::push_back(const T &value) {
    if (nn >= cap) {
        reserve(cap > 0 ? 2 * cap : 1);
    }
    v[nn++] = value;
}

template <class T>
void SLvector<T>::push_back(T &&value) {
    if (nn >= cap) {
        reserve(cap > 0 ? 2 * cap : 1);
    }
    v[nn++] = std::move(value);
}

template <class T>
template <class... Args>
void SLvector<T>::emplace_back(Args&&... args) {
    if (nn >= cap) {
        reserve(cap > 0 ? 2 * cap : 1);
    }
    new (&v[nn++]) T(std::forward<Args>(args)...);
}

template <class T>
void SLvector<T>::shrink_to_fit() {
    if (nn < cap) {
        reallocate(nn);
    }
}

template <class T>
T* SLvector<T>::begin() noexcept {
    return v.get();
}

template <class T>
const T* SLvector<T>::begin() const noexcept {
    return v.get();
}

template <class T>
T* SLvector<T>::end() noexcept {
    return v.get() + nn;
}

template <class T>
const T* SLvector<T>::end() const noexcept {
    return v.get() + nn;
}

// matrix

template <class T>
class SLMat3d {
private:
    int nn;
    int mm;
    int kk;
    std::unique_ptr<T[]> v;

public:
    SLMat3d();
    SLMat3d(int n, int m, int k);
    SLMat3d(int n, int m, int k, const T &a);
    SLMat3d(const SLMat3d &rhs); // Copy constructor
    SLMat3d(SLMat3d &&rhs) noexcept; // Move constructor
    SLMat3d & operator=(const SLMat3d &rhs); // Copy assignment
    SLMat3d & operator=(SLMat3d &&rhs) noexcept; // Move assignment
    typedef T value_type; // make T available externally
    inline T* operator[](const int i); // subscripting: pointer to layer i
    inline const T* operator[](const int i) const;
    inline int dim1() const noexcept;
    inline int dim2() const noexcept;
    inline int dim3() const noexcept;
    void resize(int newn, int newm, int newk); // resize (contents not preserved)
    void assign(int newn, int newm, int newk, const T &a); // resize and assign a constant value
    ~SLMat3d() = default;
};

// SLMat3d definitions

template <class T>
SLMat3d<T>::SLMat3d() : nn(0), mm(0), kk(0), v(nullptr) {}

template <class T>
SLMat3d<T>::SLMat3d(int n, int m, int k) : nn(n), mm(m), kk(k), v(n > 0 ? std::make_unique<T[]>(n * m * k) : nullptr) {}

template <class T>
SLMat3d<T>::SLMat3d(int n, int m, int k, const T &a) : nn(n), mm(m), kk(k), v(n > 0 ? std::make_unique<T[]>(n * m * k) : nullptr) {
    std::fill_n(v.get(), n * m * k, a);
}

template <class T>
SLMat3d<T>::SLMat3d(const SLMat3d &rhs) : nn(rhs.nn), mm(rhs.mm), kk(rhs.kk), v(nn > 0 ? std::make_unique<T[]>(nn * mm * kk) : nullptr) {
    std::copy(rhs.v.get(), rhs.v.get() + nn * mm * kk, v.get());
}

template <class T>
SLMat3d<T>::SLMat3d(SLMat3d<T> &&rhs) noexcept : nn(rhs.nn), mm(rhs.mm), kk(rhs.kk), v(std::move(rhs.v)) {
    rhs.nn = 0;
    rhs.mm = 0;
    rhs.kk = 0;
}

template <class T>
SLMat3d<T> & SLMat3d<T>::operator=(const SLMat3d<T> &rhs) {
    if (this != &rhs) {
        nn = rhs.nn;
        mm = rhs.mm;
        kk = rhs.kk;
        v = nn > 0 ? std::make_unique<T[]>(nn * mm * kk) : nullptr;
        std::copy(rhs.v.get(), rhs.v.get() + nn * mm * kk, v.get());
    }
    return *this;
}

template <class T>
SLMat3d<T> & SLMat3d<T>::operator=(SLMat3d<T> &&rhs) noexcept {
    if (this != &rhs) {
        nn = rhs.nn;
        mm = rhs.mm;
        kk = rhs.kk;
        v = std::move(rhs.v);
        rhs.nn = 0;
        rhs.mm = 0;
        rhs.kk = 0;
    }
    return *this;
}

template <class T>
inline T* SLMat3d<T>::operator[](const int i) {
#ifdef _CHECKBOUNDS_
    if (i < 0 || i >= nn) {
        throw std::out_of_range("SLMat3d subscript out of bounds");
    }
#endif
    return v.get() + i * mm * kk;
}

template <class T>
inline const T* SLMat3d<T>::operator[](const int i) const {
#ifdef _CHECKBOUNDS_
    if (i < 0 || i >= nn) {
        throw std::out_of_range("SLMat3d subscript out of bounds");
    }
#endif
    return v.get() + i * mm * kk;
}

template <class T>
inline int SLMat3d<T>::dim1() const noexcept {
    return nn;
}

template <class T>
inline int SLMat3d<T>::dim2() const noexcept {
    return mm;
}

template <class T>
inline int SLMat3d<T>::dim3() const noexcept {
    return kk;
}

template <class T>
void SLMat3d<T>::resize(int newn, int newm, int newk) {
    if (newn != nn || newm != mm || newk != kk) {
        v = newn > 0 ? std::make_unique<T[]>(newn * newm * newk) : nullptr;
        nn = newn;
        mm = newm;
        kk = newk;
    }
}

template <class T>
void SLMat3d<T>::assign(int newn, int newm, int newk, const T &a) {
    resize(newn, newm, newk);
    std::fill_n(v.get(), newn * newm * newk, a);
}

#endif // SCI_H