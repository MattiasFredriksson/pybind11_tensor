/* eigen_tensor_ext.h
* 
* Functions extending functionality of the Eigen tensor library.
* Functionality is focused on efficient slicing/chipping of tensors to Eigen matrix representations.
*/
#pragma once
#include <Eigen/Geometry>
#include "Eigen/Dense"
#include <unsupported/Eigen/CXX11/Tensor>
#include "tensor_traits.h"


#pragma region Eigen alias
namespace tensorial {

	// Tensor alias
	template<typename FP, int rank>
	using Tensor = Eigen::Tensor<FP, rank, Eigen::RowMajor>;

	template<typename FP, int rank>
	using TensorMap = Eigen::TensorMap<Eigen::Tensor<FP, rank, Eigen::RowMajor>>;
	template<typename FP, int rank>
	using TensorMapC = Eigen::TensorMap<const Eigen::Tensor<FP, rank, Eigen::RowMajor>>;

	template<int rank>
	using Tensord = Tensor<double, rank>;
	template<int rank>
	using Tensorf = Tensor<float, rank>;


	template<typename FP = double, int dim = Eigen::Dynamic>
	using Vector = Eigen::Matrix<FP, dim, 1, Eigen::ColMajor>;
	template<typename FP = double>
	using Vector2 = Eigen::Matrix<FP, 2, 1, Eigen::ColMajor>;
	template<typename FP = double>
	using Vector3 = Eigen::Matrix<FP, 3, 1, Eigen::ColMajor>;
	template<typename FP = double>
	using Vector4 = Eigen::Matrix<FP, 4, 1, Eigen::ColMajor>;

	template<typename FP = double>
	using Matrix3 = Eigen::Matrix<FP, 3, 3, Eigen::RowMajor>;
	template<typename FP = double>
	using Matrix4 = Eigen::Matrix<FP, 4, 4, Eigen::RowMajor>;

	template<typename FP = double>
	using MatrixN2 = Eigen::Matrix<FP, Eigen::Dynamic, 2, Eigen::RowMajor>;
	template<typename FP = double>
	using MatrixN3 = Eigen::Matrix<FP, Eigen::Dynamic, 3, Eigen::RowMajor>;
	template<typename FP = double>
	using MatrixN4 = Eigen::Matrix<FP, Eigen::Dynamic, 4, Eigen::RowMajor>;
	template<typename FP = double, int rows = Eigen::Dynamic, int cols = Eigen::Dynamic>
	using MatrixNN = Eigen::Matrix<FP, rows, cols, Eigen::RowMajor>;
	template<typename FP = double, int width = Eigen::Dynamic>
	using Matrix = MatrixNN<FP, width, width>;



	template<typename FP = double>
	using MatrixMapN2 = Eigen::Map<MatrixN2<FP>>;
	template<typename FP = double>
	using MatrixMapN3 = Eigen::Map<MatrixN3<FP>>;
	template<typename FP = double>
	using MatrixMapN4 = Eigen::Map<MatrixN4<FP>>;


	template<typename FP = double>
	using MapN2 = Eigen::Map<Eigen::Matrix<FP, Eigen::Dynamic, 2, Eigen::RowMajor>>;
	template<typename FP = double>
	using MapNNC = Eigen::Map<const Eigen::Matrix<FP, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

	template<typename FP = double>
	using RefN2 = Eigen::Ref<Eigen::Matrix<FP, Eigen::Dynamic, 2, Eigen::RowMajor>>;
	template<typename FP = double>
	using RefNNC = Eigen::Ref<const Eigen::Matrix<FP, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

	/* Note:
	Eigen quaternions are ordered [x,y,z,w] in memory with the scalar last, this is the same order as Scipy.Rotatin module.
	*/
	template<typename FP = double>
	using Quat = Eigen::Quaternion<FP>;
#pragma endregion

#pragma region slice

	/**
	* <summary>Convert dense tensor to matrix.</summary>
	*/
	template<typename Scalar, int rank, typename sizeType>
	auto tensor2matrix(const Eigen::Tensor<Scalar, rank>& tensor)
	{
		return Eigen::Map<const MatrixType<Scalar>>(tensor.data(), tensor.dimension(0), tensor.dimension(1));
	}
	/**
	* <summary>Convert matrix map to tensor.</summary>
	*/
	template<typename Scalar, int Rows, int Cols, int Major>
	auto matrix2tensor(const Eigen::Map<Eigen::Matrix<Scalar, Rows, Cols, Major>>& matrix)
	{
		return Eigen::TensorMap<Eigen::Tensor<Scalar, 2, Major>>(matrix.data(), matrix.rows(), matrix.cols());
	}
	/**
	* <summary>Convert dense matrix to tensor.</summary>
	*/
	template<typename Scalar, int Rows, int Cols, int Major>
	auto matrix2tensor(Eigen::Matrix<Scalar, Rows, Cols, Major>& matrix)
	{
		return Eigen::TensorMap<Eigen::Tensor<Scalar, 2, Major>>(matrix.data(), matrix.rows(), matrix.cols());
	}
	/**
	* <summary>Convert vector map to tensor.</summary>
	*/
	template<typename Scalar, int Rows, int Cols, int Major>
	auto vector2tensor(const Eigen::Map<Eigen::Matrix<Scalar, Rows, Cols, Major>>& vector)
	{
		return Eigen::TensorMap<Eigen::Tensor<Scalar, 1, Eigen::RowMajor>>(vector.data(), vector.size());
	}
	/**
	* <summary>Convert dense vector to tensor.</summary>
	*/
	template<typename Scalar, int Rows, int Cols, int Major>
	auto vector2tensor(Eigen::Matrix<Scalar, Rows, Cols, Major>& vector)
	{
		return Eigen::TensorMap<Eigen::Tensor<Scalar, 1, Eigen::RowMajor>>(vector.data(), vector.size());
	}

#pragma region Helpers

	/*	Unpacks variadic arg -> std::array of type T and size N.
	*/
	template<class T, size_t N, class ... Values>
	void assign_values(std::array<T, N>& arr, Values... vals) {
		static_assert(N == sizeof...(vals));	// assert variadic count match array size N
		int j = 0;
		for (auto i : std::initializer_list< std::common_type_t<Values...> >{ vals... })
			arr[j++] = i;
	}

	/*	Index offset for the N first dimensions for row/column tensors. (Column untested).
	*/
	template <typename TensorType, typename... Ix>
	std::int64_t tensor_offset(TensorType& t, Ix... index) {
		constexpr bool isrow = is_eigen_row_major_tensor<TensorType>::value;
		constexpr std::int64_t N = std::int64_t{ sizeof...(Ix) };
		constexpr std::int64_t incr = isrow ? -1 : 1;
		constexpr std::int64_t beg = isrow ? TensorType::NumIndices - 1 : 0;
		constexpr std::int64_t end = isrow ? N - 1 : 0;
		constexpr std::int64_t last = isrow ? -1 : N;
		static_assert(N <= TensorType::NumIndices,
			"Invalid number of indices");// , expected at most" + std::to_string(TensorType::NumIndices));
		std::int64_t cum_stride = 1;
		for (std::int64_t i = beg; i != end; i += incr) // Only iterated for row tensors.
			cum_stride *= t.dimension(i);

		// Make variadic arg -> array
		std::array<std::int64_t, N> inds;
		assign_values(inds, index...);

		std::int64_t off = 0;
		for (std::int64_t i = end; i != last; i += incr) {
			off += inds[i] * cum_stride;
			cum_stride *= t.dimension(i);
		}
		return off;
	}

#pragma endregion


using EigenStride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
using InnerTStride = Eigen::Stride<1, Eigen::Dynamic>;

#pragma region slice row major column vector

	/**
	* <summary>Get a column vector slice (subtensor) from a row major tensor.</summary>
	* <param name="tensor">Row major tensor of rank N.</param>
	* <param name="slice_offsets"> Offset indices for the subtensor within the first N-1 rank dimensions.</param>
	* <returns>A mapped matrix view of the subtensor slice.</returns>
	*/
	template<typename TensorType, typename... Ix, std::enable_if_t<std::conjunction<
		is_eigen_row_major_tensor<TensorType>,
		std::negation<is_eigen_mutable_tensor<TensorType>>>::value,
		int> = 0>
		Eigen::Map<const Eigen::Matrix<typename TensorType::Scalar, Eigen::Dynamic, 1, Eigen::ColMajor>>
		slice_vector(TensorType& tensor, typename Ix... slice_offset) { //slice_offsets[TensorType::Dimensions - 2]
		static_assert(TensorType::Layout == Eigen::RowMajor, "Invalid tensor layout type, expected tensor to be row major.");
		static_assert(std::size_t{ sizeof...(Ix) } == TensorType::NumIndices - 1,
			"Incorrect number of indices passed to slice function.");//, expected " + std::to_string(TensorType::NumIndices - 1));
		Eigen::Index d1 = tensor.dimension(TensorType::NumIndices - 1);
		Eigen::Stride<0, 0> stride;
		// Calc. offset
		Eigen::Index offset = tensor_offset(tensor, slice_offset...);

		return Eigen::Map<const Eigen::Matrix<TensorType::Scalar, Eigen::Dynamic, 1, Eigen::ColMajor>>(
			tensor.data() + offset,
			d1,
			1,
			stride);
	}
	/**
	* <summary>Get a column vector slice (subtensor) from a row major tensor.</summary>
	* <param name="tensor"> Row major tensor of rank N.</param>
	* <param name="slice_offsets"> Offset indices for the subtensor within the first N-2 rank dimensions.</param>
	* <returns>A mapped matrix view of the subtensor slice.</returns>
	*/
	template<typename TensorType, typename... Ix, std::enable_if_t<std::conjunction<
		is_eigen_row_major_tensor<TensorType>,
		is_eigen_mutable_tensor<TensorType>>::value,
		int> = 0>
		Eigen::Map<Eigen::Matrix<typename TensorType::Scalar, Eigen::Dynamic, 1, Eigen::ColMajor>>
		slice_vector(TensorType& tensor, typename Ix... slice_offset) {
		// Return const. version
		return *reinterpret_cast<Eigen::Map<Eigen::Matrix<typename TensorType::Scalar, Eigen::Dynamic, 1, Eigen::ColMajor>>*>(
			&slice_vector(tensor, slice_offset...));
	}

#pragma endregion 

#pragma region slice matrix

#pragma region slice row major matrix


	/**
	* <summary>Get a row major matrix slice (subtensor) from a row major tensor.</summary>
	* <param name="tensor"> Row major tensor of rank N.</param>
	* <param name="slice_offsets"> Offset indices for the subtensor within the first N-2 rank dimensions.</param>
	* <returns>A mapped matrix view of the subtensor slice.</returns>
	*/
	template<typename TensorType, typename... Ix, std::enable_if_t<std::conjunction<
		is_eigen_row_major_tensor<TensorType>,
		std::negation<is_eigen_mutable_tensor<TensorType>>>::value,
		int> = 0>
		Eigen::Map<const Eigen::Matrix<typename TensorType::Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::RowMajor, EigenStride>
		slice_matrix(TensorType& tensor, typename Ix... slice_offset) { //slice_offsets[TensorType::Dimensions - 2]
		static_assert(TensorType::Layout == Eigen::RowMajor, "Invalid tensor layout type, expected tensor to be row major.");
		static_assert(std::size_t{ sizeof...(Ix) } == TensorType::NumIndices - 2,
			"Incorrect number of indices passed to slice function.");//, expected " + std::to_string(TensorType::NumIndices - 2));
		Eigen::Index d1 = tensor.dimension(TensorType::NumIndices - 2);
		Eigen::Index d2 = tensor.dimension(TensorType::NumIndices - 1);
		EigenStride stride(d2, 1);
		// Calc. offset
		Eigen::Index offset = tensor_offset(tensor, slice_offset...);

		return Eigen::Map<const Eigen::Matrix<TensorType::Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::RowMajor, EigenStride>(
			tensor.data() + offset,
			d1,
			d2,
			stride);
	}
	/**
	* <summary>Get a row major matrix slice (subtensor) from a row major tensor.</summary>
	* <param name="tensor"> Row major tensor of rank N.</param>
	* <param name="slice_offsets"> Offset indices for the subtensor within the first N-2 rank dimensions.</param>
	* <returns>A mapped matrix view of the subtensor slice.</returns>
	*/
	template<typename TensorType, typename... Ix, std::enable_if_t<std::conjunction<
		is_eigen_row_major_tensor<TensorType>,
		is_eigen_mutable_tensor<TensorType>>::value,
		int> = 0>
		Eigen::Map<Eigen::Matrix<typename TensorType::Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::RowMajor, EigenStride>
		slice_matrix(TensorType& tensor, typename Ix... slice_offset) {
		// Return const. version
		return *reinterpret_cast<Eigen::Map<Eigen::Matrix<typename TensorType::Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::RowMajor, Eigen::InnerStride<>>*>(
			&slice_matrix(tensor, slice_offset...));
	}




	/**
	* <summary>Get a column major matrix slice (subtensor) from a row major tensor.</summary>
	* <param name="tensor"> Row major tensor of rank N.</param>
	* <param name="slice_offsets"> Offset indices for the subtensor within the first N-2 rank dimensions.</param>
	* <returns>A mapped matrix view of the subtensor slice.</returns>
	*/
	template<typename TensorType, typename... Ix, std::enable_if_t<std::conjunction<
		is_eigen_row_major_tensor<TensorType>,
		std::negation<is_eigen_mutable_tensor<TensorType>>>::value,
		int> = 0>
		Eigen::Map<const Eigen::Matrix<typename TensorType::Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, Eigen::ColMajor, EigenStride>
		slice_matrix_col(TensorType& tensor, typename Ix... slice_offset) { //slice_offsets[TensorType::Dimensions - 2]
		static_assert(TensorType::Layout == Eigen::RowMajor, "Invalid tensor layout type, expected tensor to be row major.");
		static_assert(std::size_t{ sizeof...(Ix) } == TensorType::NumIndices - 2,
			"Invalid number of indices");//, expected " + std::to_string(TensorType::NumIndices - 2));
		Eigen::Index d1 = tensor.dimension(TensorType::NumIndices - 2);
		Eigen::Index d2 = tensor.dimension(TensorType::NumIndices - 1);
		EigenStride stride(1, d1);
		// Calc. offset
		Eigen::Index offset = tensor_offset(tensor, slice_offset...);

		return Eigen::Map<const Eigen::Matrix<TensorType::Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, Eigen::ColMajor, EigenStride>(
			tensor.data() + offset,
			d1,
			d2,
			stride);
	}
	/**
	* <summary>Get a a column major matrix slice (subtensor) from a row major tensor.</summary>
	* <param name="tensor"> Row major tensor of rank N.</param>
	* <param name="slice_offsets"> Offset indices for the subtensor within the first N-2 rank dimensions.</param>
	* <returns>A mapped matrix view of the subtensor slice.</returns>
	*/
	template<typename TensorType, typename... Ix, std::enable_if_t<std::conjunction<
		is_eigen_row_major_tensor<TensorType>,
		is_eigen_mutable_tensor<TensorType>>::value,
		int> = 0>
		Eigen::Map<Eigen::Matrix<typename TensorType::Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, Eigen::ColMajor, EigenStride>
		slice_matrix_col(TensorType& tensor, typename Ix... slice_offset) {
		// Return const. version
		return *reinterpret_cast<Eigen::Map<Eigen::Matrix<typename TensorType::Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, Eigen::ColMajor, Eigen::InnerStride<>>*>(
			&slice_matrix_col(tensor, slice_offset...));
	}

#pragma endregion

#pragma region slice column major matrix

	/**
	* const *
	* <summary>Get a matrix slice from a column major tensor of rank 3.</summary>
	* <param name="tensor"> Column major tensor of rank 3.</param>
	* <param name="index"> Index of the (matrix) sub-tensor within the tensor.</param>
	* <returns>A matrix map of the sub-tensor slice.</returns>
	*/
	template<typename TensorType, std::enable_if_t<std::conjunction<
		is_eigen_col_major_tensor<TensorType>,
		std::negation<is_eigen_mutable_tensor<TensorType>>>::value,
		int> = 0>
		Eigen::Map<const Eigen::Matrix<typename TensorType::Scalar, Eigen::Dynamic, Eigen::Dynamic>, Eigen::ColMajor, Eigen::InnerStride<>>
		slice_matrix_col(TensorType tensor, Eigen::Index index) {
		static_assert(TensorType::NumIndices == 3, "This function currently only support 3-dim tensors.");
		static_assert(TensorType::Layout == Eigen::ColMajor, "Invalid tensor layout type, expected tensor to be column major.");
		auto d1 = tensor.dimension(TensorType::NumIndices - 2);
		auto d2 = tensor.dimension(TensorType::NumIndices - 1);
		Eigen::InnerStride<> stride(tensor.dimension(0));
		return Eigen::Map<const Eigen::Matrix<TensorType::Scalar, Eigen::Dynamic, Eigen::Dynamic>, Eigen::ColMajor, Eigen::InnerStride<>>(
			tensor.data() + index,
			d1,
			d2,
			stride);
	}

	/**
	* <summary>Get a matrix slice from a column major tensor of rank 3.</summary>
	* <param name="tensor"> Column major tensor of rank 3.</param>
	* <param name="index"> Index of the (matrix) sub-tensor within the tensor.</param>
	* <returns>A matrix map of the sub-tensor slice.</returns>
	*/
	template<typename TensorType, std::enable_if_t<std::conjunction<
		is_eigen_col_major_tensor<TensorType>,
		is_eigen_mutable_tensor<TensorType>>::value,
		int> = 0>
		Eigen::Map<Eigen::Matrix<typename TensorType::Scalar, Eigen::Dynamic, Eigen::Dynamic>, Eigen::ColMajor, Eigen::InnerStride<>>
		slice_matrix_col(TensorType tensor, Eigen::Index index) {
		// Non-const version
		return *reinterpret_cast<Eigen::Map<Eigen::Matrix<typename TensorType::Scalar, Eigen::Dynamic, Eigen::Dynamic>, Eigen::ColMajor, Eigen::InnerStride<>>*>(&slice_matrix(tensor, index));
	}

#pragma endregion

#pragma endregion
#pragma endregion

#pragma region tensoriterator

	/**
* <summary>Iterator providing access to subtensors along the first rank dimension of the tensor.</summary>
*/
	template <typename TensorType>
	struct tensoriterator {
	public:
		// Todo: implement begin() end()
		using Scalar = typename TensorType::Scalar;
		static const int Rank = TensorType::NumIndices;

		static const bool is_const_tensor = std::is_const<typename eigen_nested_type<TensorType>::type>::value;

		// Return Tensor
		using ITensor = typename std::conditional<
			std::is_const<typename eigen_nested_type<TensorType>::type>::value,
			typename TensorMapC<Scalar, Rank - 1>,
			typename TensorMap<Scalar, Rank - 1>>::type;

		// Nested dense type
		using DenseTensorType = typename std::conditional<
			std::is_const<typename eigen_nested_type<TensorType>::type>::value,
			typename const Tensor<Scalar, Rank>,
			typename Tensor<Scalar, Rank>>::type;

	private:
		/* Reference tensor. */
		TensorType m_tensor;
		Eigen::Index m_stride;
		Eigen::array<Eigen::DenseIndex, Rank - 1> m_shape;
	public:

		void init() {
			// Determine shape
			for (size_t i = 1; i < Rank; i++) {
				Eigen::Index d = m_tensor.dimension(i);
				this->m_shape[i - 1] = d;
				this->m_stride *= d;
			}
		}

		/*
		* <summary>RH constructor == non-reference, non-dense constructor. </summary>
		*/
		tensoriterator(TensorType&& tensor)
			: m_tensor(TensorType(tensor.data(), tensor.dimensions())), m_stride(1) {
			init();
		}
		/*
		* <summary>LH constructor == reference constructors (dense and non-dense). </summary>
		*/
		tensoriterator(TensorType& tensor)
			: m_tensor(TensorType(tensor.data(), tensor.dimensions())), m_stride(1) {
			init();
		}
		tensoriterator(DenseTensorType& tensor)
			: m_tensor(TensorType(tensor.data(), tensor.dimensions())), m_stride(1) {
			init();
		}

		/*
		* <summary>Get the size of the iterated dimension. TODO: RENAME count</summary>
		*/
		size_t size() { return m_tensor.dimension(0); }

		const Eigen::array<Eigen::DenseIndex, Rank - 1>& shape() const { return this->m_shape; }
		Eigen::Index stride() const { return this->m_stride; }

		struct subtensor {
			/*	Wrapper for TensorMap where the struct is replaced on assignment
				instead of overwriting the data buffer.

				i.e. default behavior for assignment is replacement of the subtensor variable:

				new(&this->tensor) Tensor<FP, rank - 1>(other.tensor);

				----

				To overwrite the tensor use the iterator:

				tensoriterator iter(..);
				iter(i) = iter(j);

				or apply to the tensor argument:

				subtensor subt;
				subt.tensor() = some_tensor;
				subt.ref() = other_subt;

			*/
		private:
			/* View of reference tensor. */
			ITensor m_tensor;

		public:

			subtensor(ITensor tensor)
				: m_tensor(tensor) { }

			operator ITensor& () { return this->m_tensor; }
			operator const ITensor& () const { return this->m_tensor; }
			ITensor& ref() { return this->m_tensor; }
			const ITensor& ref() const { return this->m_tensor; }
			ITensor& tensor() { return this->m_tensor; }
			const ITensor& tensor() const { return this->m_tensor; }

#pragma region subtensor: copy & assign

			subtensor(const subtensor& o)
				: m_tensor(o.tensor) { }
			subtensor(subtensor&& o)
				: m_tensor(std::move(o.tensor)) {	}

			subtensor& operator=(subtensor& other) = delete;
			subtensor& operator=(const subtensor& other) noexcept
			{
				if (&other == this)
					return *this;
				new(&this->m_tensor) ITensor(other.m_tensor);
				return *this;
			}
			subtensor& operator=(subtensor&& other) noexcept
			{
				if (&other == this)
					return *this;
				new(&this->m_tensor) ITensor(other.m_tensor);
				return *this;
			}

#pragma endregion

		};

		/**
		* <summary>Get a replaceable view of the indexed subtensor in the first rank dimension of the reference.</summary>
		* <param name="index"> Index of the subtensor within the first rank dimension.</param>
		* <returns>Replaceable view of the subtensor. The specialized view wrapper allow assigning a replacement of a subtensor variable without overwriting the underlying tensor.</returns>
		*/
		tensoriterator::subtensor operator[](const Eigen::Index index) const {
			return tensoriterator::subtensor(this->operator()(index));
		}
		/**
		* <summary>Get a view of the subtensor in the first rank dimension of the reference tensor.</summary>
		* <param name="index">Index of the subtensor within the first rank dimension.</param>
		* <returns>View of the subtensor.</returns>
		*/
		ITensor operator()(const Eigen::Index index) const {
			return ITensor(m_tensor.data() + index * this->m_stride, this->m_shape);
		}

#pragma region copy & assign

		tensoriterator(const tensoriterator& o)
			: m_tensor(o.tensor), m_stride(o.m_stride), m_shape(o.m_shape) {
		}
		tensoriterator(tensoriterator&& o)
			: m_tensor(std::move(o.m_tensor)), m_stride(o.m_stride), m_shape(std::move(o.m_shape)) {
		}

		tensoriterator& operator=(tensoriterator& other) = delete;
		tensoriterator& operator=(const tensoriterator& other) noexcept
		{
			if (&other == this)
				return *this;
			new(&this->m_tensor) ITensor(other.m_tensor);
			this->m_stride = other.m_stride;
			this->m_shape = other.m_shape;
			return *this;
		}
		tensoriterator& operator=(tensoriterator&& other) noexcept
		{
			if (&other == this)
				return *this;
			this->m_tensor = std::move(other.m_tensor);
			this->m_stride = other.m_stride;
			this->m_shape = std::move(other.m_shape);
			return *this;
		}

#pragma endregion

	};





#pragma endregion


#pragma region Helpers

	/*
	*	Verify the number of dimensions in given rank dimension match.
	*/
	template <typename T0>
	void assert_rank_dimension(const T0& t, int rank, int dimension) {

		if (t.dimension(rank) != dimension) {
			std::ostringstream buf;
			buf << "Expected tensor argument to be of dimension " << dimension << " in rank dimension " << rank << ", was " << t.dimension(rank);
			buf << " for tensor argument with shape (" << t.dimensions() << ").";
			throw std::invalid_argument(buf.str());
		}
	}

	/*
	*	Verify the number of dimensions within the given rank of the tensor arguments match.
	*/
	template <typename T0, typename T1>
	void assert_rank_dimension_match(const T0& t0, const T1& t1, int rank) {

		if (t0.dimension(rank) != t1.dimension(rank)) {
			std::ostringstream buf;
			buf << "Mismatch in dimensions within rank " << rank << " of the tensor arguments. First argument was of shape: (";
			buf << t0.dimensions() << "), second argument of shape (" << t1.dimensions() << ").";
			throw std::invalid_argument(buf.str());
		}
	}
	/*
	*	Verify the number of dimensions within the given rank of the tensor arguments match.
	*/
	template <typename T0, typename T1>
	void assert_rank_dimension_match(const T0& t0, const T1& t1, int rank0, int rank1) {

		if (t0.dimension(rank0) != t1.dimension(rank1)) {
			std::ostringstream buf;
			buf << "Mismatch in rank dimensions for rank " << rank0 << " in the first tensor argument and rank ";
			buf << rank1 << " in the second. First argument was of shape: (";
			buf << t0.dimensions() << "), second argument of shape (" << t1.dimensions() << ").";
			throw std::invalid_argument(buf.str());
		}
	}

	/*	
	*	Eigen::array operators
	*/

    template <typename TIndex, int rank>
    std::ostream& operator <<(std::ostream& stream, const Eigen::array<TIndex, rank>& arr) {
		int i = 0;
        for (; i < rank - 1; i++)
            stream << arr[i] << ", ";
		stream << arr[i];
        return stream;
    }
    // Divide Eigen::array
    template <typename Index, int rank>
    Eigen::array<Index, rank> operator/(const Eigen::array<Index, rank>& numerator, const Eigen::array<Index, rank>& denom) {
        Eigen::array<Index, rank> res;
        for (int i = 0; i < rank; i++)
            res[i] = numerator[i] / denom[i];
        return res;
    }
    // Mult Eigen::array
    template <typename Index, int rank>
    Eigen::array<Index, rank> operator*(const Eigen::array<Index, rank>& numerator, const Eigen::array<Index, rank>& denom) {
        Eigen::array<Index, rank> res;
        for (int i = 0; i < rank; i++)
            res[i] = numerator[i] * denom[i];
        return res;
    }

#pragma endregion

}