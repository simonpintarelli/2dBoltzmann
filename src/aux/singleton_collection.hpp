#pragma once

#include <memory>
#include <mutex>

namespace boltzmann {

template <typename T, typename V>
class SingletonCollection
{
 public:
  using value_t = V;

 public:
  template <class... U>
  static value_t& GetInstance(U&&... u);

 private:
  static std::unique_ptr<T> m_instance;
  static std::once_flag m_onceFlag;
  SingletonCollection(void) {}
  friend T;
};

template <typename T,
          typename V>
std::unique_ptr<T> SingletonCollection<T,V>::m_instance;

template <typename T,
          typename V>
std::once_flag SingletonCollection<T, V>::m_onceFlag;

template <typename T,
          typename V>
template <class... U>
inline typename SingletonCollection<T, V>::value_t&
SingletonCollection<T, V>::GetInstance(U&&... u)
{
  std::call_once(m_onceFlag, [] { m_instance.reset(new T); });
  return m_instance.get()->make(std::forward<U>(u)...);
}


}  // boltzmann
