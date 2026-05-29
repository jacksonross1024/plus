#pragma once

#include <cstdint>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace poisson_npy {

template <typename T>
struct Array {
  std::vector<std::size_t> shape;
  std::vector<T> data;
};

struct Header {
  std::string descr;
  bool fortran_order = false;
  std::vector<std::size_t> shape;
};

inline std::string trim(const std::string& text) {
  const auto first = text.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return "";
  }
  const auto last = text.find_last_not_of(" \t\r\n");
  return text.substr(first, last - first + 1);
}

inline std::size_t checked_product(const std::vector<std::size_t>& shape) {
  return std::accumulate(
      shape.begin(), shape.end(), std::size_t{1},
      [](std::size_t lhs, std::size_t rhs) -> std::size_t {
        if (rhs != 0 && lhs > static_cast<std::size_t>(-1) / rhs) {
          throw std::runtime_error("npy shape product overflow");
        }
        return lhs * rhs;
      });
}

inline std::string extract_after_key(const std::string& header, const std::string& key) {
  const std::string token = "'" + key + "':";
  const auto token_pos = header.find(token);
  if (token_pos == std::string::npos) {
    throw std::runtime_error("missing npy header key: " + key);
  }
  return header.substr(token_pos + token.size());
}

inline std::string parse_descr(const std::string& header) {
  const auto tail = extract_after_key(header, "descr");
  const auto begin = tail.find('\'');
  const auto end = tail.find('\'', begin + 1);
  if (begin == std::string::npos || end == std::string::npos) {
    throw std::runtime_error("malformed npy descr field");
  }
  return tail.substr(begin + 1, end - begin - 1);
}

inline bool parse_fortran_order(const std::string& header) {
  const auto tail = trim(extract_after_key(header, "fortran_order"));
  if (tail.rfind("True", 0) == 0) {
    return true;
  }
  if (tail.rfind("False", 0) == 0) {
    return false;
  }
  throw std::runtime_error("malformed npy fortran_order field");
}

inline std::vector<std::size_t> parse_shape(const std::string& header) {
  const auto tail = extract_after_key(header, "shape");
  const auto lparen = tail.find('(');
  const auto rparen = tail.find(')', lparen);
  if (lparen == std::string::npos || rparen == std::string::npos) {
    throw std::runtime_error("malformed npy shape field");
  }

  std::vector<std::size_t> shape;
  std::stringstream ss(tail.substr(lparen + 1, rparen - lparen - 1));
  std::string item;
  while (std::getline(ss, item, ',')) {
    item = trim(item);
    if (!item.empty()) {
      shape.push_back(static_cast<std::size_t>(std::stoull(item)));
    }
  }
  return shape;
}

inline Header read_header(std::istream& in) {
  char magic[6];
  in.read(magic, sizeof(magic));
  if (!in || std::string(magic, sizeof(magic)) != "\x93NUMPY") {
    throw std::runtime_error("invalid npy magic header");
  }

  char version[2];
  in.read(version, sizeof(version));
  if (!in) {
    throw std::runtime_error("unable to read npy version");
  }

  std::uint32_t header_len = 0;
  if (version[0] == 1) {
    std::uint16_t v1_len = 0;
    in.read(reinterpret_cast<char*>(&v1_len), sizeof(v1_len));
    header_len = v1_len;
  } else if (version[0] == 2 || version[0] == 3) {
    in.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
  } else {
    throw std::runtime_error("unsupported npy version");
  }

  std::string header_text(header_len, '\0');
  in.read(header_text.data(), static_cast<std::streamsize>(header_text.size()));
  if (!in) {
    throw std::runtime_error("unable to read npy header body");
  }

  return {parse_descr(header_text), parse_fortran_order(header_text), parse_shape(header_text)};
}

template <typename T>
inline bool descr_matches(const std::string& descr) {
  if constexpr (std::is_same_v<T, float>) {
    return descr == "<f4";
  } else if constexpr (std::is_same_v<T, std::int8_t>) {
    return descr == "|i1" || descr == "<i1";
  }
  return false;
}

template <typename T>
inline Array<T> read_npy_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("unable to open npy file: " + path);
  }

  const Header header = read_header(in);
  if (header.fortran_order) {
    throw std::runtime_error("Fortran-order npy arrays are not supported: " + path);
  }
  if (!descr_matches<T>(header.descr)) {
    throw std::runtime_error("unexpected npy dtype in " + path + ": " + header.descr);
  }

  Array<T> array;
  array.shape = header.shape;
  array.data.resize(checked_product(array.shape));
  in.read(reinterpret_cast<char*>(array.data.data()),
          static_cast<std::streamsize>(array.data.size() * sizeof(T)));
  if (!in) {
    throw std::runtime_error("unable to read npy payload: " + path);
  }
  return array;
}

}  // namespace poisson_npy
