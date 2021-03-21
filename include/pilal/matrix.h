/*
  This file is part of C++lex, a project by Tommaso Urli.

  C++lex is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  C++lex is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with C++lex.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef PILAL_MATRIX_H
#define PILAL_MATRIX_H

#include "pilal.h"
#include <utility>	// std::pair
#include <vector>	// std::vector
#include <iostream> // std::string
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <cmath>

namespace pilal {

    /** Enum type that describes the two types of data used to perform
	permutations: permutation matrices or permutation vectors (explained
	later).  */
    enum PermutationFormat {
        PF_MATRIX,
        PF_VECTOR
    };

    /** Enum type that describes the shape of the matrix to optimize
        matrix inversion for special matrices (triangular, permutation
        or generic). */

    enum MatrixType {
        MT_GENERIC,
        MT_TRIANGULAR_UPPER,
        MT_TRIANGULAR_LOWER,
        MT_PERMUTATION
    };

    /** Forward declaration for anonymous Matrix. */
    template <typename Scalar> class AnonymousMatrix;

    /** Represents a matrix. */
    template <typename Scalar>
    class Matrix {

        /** Provide access to anonymous matrices. */
        friend class AnonymousMatrix<Scalar>;

    public:

        /** Default constructor. */
        Matrix() :
            lu_up_to_date(false),
            determinant_up_to_date(false),
            inverse_up_to_date(false),
            values(new storage(0)),
            rows(0),
            columns(0) {
        }


        /** Constructor which accepts a string of values and builds a single row matrix. */
        Matrix(char const * values) :
            lu_up_to_date(false),
            determinant_up_to_date(false),
            inverse_up_to_date(false) {

            int chunks = 0;
            long double ignore;
            std::stringstream buffer(values);
            while (!buffer.eof()) {
                buffer >> ignore;
                ++chunks;
            }

            this->rows = 1;
            this->columns = chunks;
            this->values = new storage(chunks);
            set_values(values);

        }

        /** Constructor which builds a square matrix of dimension n. */
        Matrix(int n) :
            lu_up_to_date(false),
            determinant_up_to_date(false),
            inverse_up_to_date(false),
            values(new storage(n * n)),
            rows(n),
            columns(n) {
        }


        /** Constructor which builds a square matrix of dimension n, and initializes each element to v. */
	Matrix(int n, Scalar v) :
	    lu_up_to_date(false),
	    determinant_up_to_date(false),
	    inverse_up_to_date(false),
            values(new storage(n * n, v)),
            rows(n),
            columns(n) {
	}

        /** Constructor which builds a r x c matrix. */
        Matrix(int r, int c) :
	    lu_up_to_date(false),
	    determinant_up_to_date(false),
	    inverse_up_to_date(false),
            values(new storage(r * c)),
            rows(r),
            columns(c) {
	}

        /** Constructor which builds a r x c matrix, and initializes each element to v. */
        Matrix(int r, int c, Scalar v) :
	    lu_up_to_date(false),
	    determinant_up_to_date(false),
	    inverse_up_to_date(false),
            values(new storage(r * c, v)),
            rows(r),
            columns(c) {
	}

        /** Construct an r x c zero matrix. This provides the same interface to the equivalent
            Eigen3 function. */
        static Matrix<Scalar> Zero(int r, int c) {
            return Matrix<Scalar> (r, c, 0);
        }

        /** Copy constructor. */
        Matrix(Matrix<Scalar> const& m) :
             lu_up_to_date(false),
             determinant_up_to_date(false),
             inverse_up_to_date(false),
             values(new storage(*m.values)),
             rows(m.rows),
             columns(m.columns) {
	}

        /** Create a matrix from an anonymous matrix (copies data pointer). */
        Matrix(AnonymousMatrix<Scalar> m)  :
	    lu_up_to_date(false),
	    determinant_up_to_date(false),
	    inverse_up_to_date(false),
	    values(new storage(0)),
            rows(m.rows),
            columns(m.columns) {

            swap( values->contents, m.values->contents );
        }

        virtual ~Matrix() {
	    delete values;
	}


        /** Accessor for data elements, can be modified to support caching. */
        class storage_accessor {

        public:

            /** Constructor, accepts a reference to a value and a parent matrix. */
            storage_accessor(Scalar& dest, Matrix<Scalar>& parent);

            /** Implicit cast operator, used in reading. */
            operator Scalar const& () const;                           // Reading


            storage_accessor& operator=(storage_accessor& new_value);       // Copying
            storage_accessor& operator=(Scalar const& new_value);      // Writing

        private:

            /** Reference to real value. */
            Scalar& dest;

            /** Owner. */
            Matrix& parent;
        };

        /*=========================================================
          Query and log operators
          =========================================================*/

        /** Dimension of the matrix. */
        virtual std::pair<int,int> dim() const;

        /** Prints the matrix with a name for debug. */
        void log(std::string name) const;

        /** Prints the matrix in a format compatible with octave. */
        void logtave(std::string name) const;

        /** Is the matrix square? */
        bool is_square() const;

        /** Is the matrix an identity (with tolerance value)? */
        bool is_identity(Scalar tol) const;

        /** How much storage space does the matrix uses? */
        double space() const;

        /** Compare two values (with tolerance). */
        bool more_equal_than (Scalar value, Scalar tol) const;

        /** Compare two values (with tolerance). */
        bool less_equal_than (Scalar value, Scalar tol) const;

        /*=========================================================
          Mathematical and manipulation operators
          =========================================================*/

        /** Subtracts a matrix and an anonymous matrix, generates an anonymous matrix. */
        AnonymousMatrix<Scalar> operator- (AnonymousMatrix<Scalar> m) const;

        /** Adds a matrix to an anonymous matrix, generates an anonymous matrix. */
        AnonymousMatrix<Scalar> operator+ (AnonymousMatrix<Scalar> m) const;

        /** Swaps columns r and w in the matrix. */
        void swap_columns(int r, int w);

        /** Swaps rows r and w in the matrix. */
        void swap_rows(int r, int w);

        /** Writes the value of the determinant. */
        void set_determinant(Scalar d);

        /** Reset the matrix to an identity. */
        void set_identity();

        /** Transposes the matrix. */
        void transpose();

        /** Set the matrix size t r x c.*/
        void resize(int r, int c);

        /** Fills the matrix with zeroes. */
        void empty();

        /** Alias for empty() for compatibility with equivalent Eigen3 function. */
        void setZero() {
            this->empty();
        }

        /** Fills a row of the matrix with elements in string row. */
        void set_row(int i, char const* row);

        /** Fills a column of the matrix with elements in string column. */
        void set_column(int j, char const* column);

        /** Fills the matrix with the elements in values. */
        void set_values(char const* values);

        /** Retrieves the determinant of the matrix.  */
        Scalar determinant() const;

        /*=========================================================
          Factorizations and inverses
          =========================================================*/

        /** Performs a LU factorization and stores the l, u matrices and permutation data respective
            into l, u and p. The third parameter determines the format of permutation data p (vector or matrix). */
        void get_lupp(Matrix<Scalar>& l, Matrix<Scalar>& u,
                      Matrix<Scalar>& p, PermutationFormat pf) const;

        /** Get inverse of the matrix and stores it into inverse. */
        void get_inverse(Matrix<Scalar>& inverse) const;

        /** Get inverse of the matrix and stores it into a matrix of type mt. */
        void get_inverse(Matrix<Scalar>& inverse, MatrixType mt) const;

        /** Updates inverse of the matrix after a column has changed. */
        static void get_inverse_with_column(Matrix<Scalar> const& old_inverse,
                                            Matrix<Scalar> const& new_column,
                                            int column_index,
                                            Matrix<Scalar>& new_inverse);

        /** Solves the linear problem represented by the matrix using the data vector b. */
        void solve(Matrix& x, Matrix const& b) const;

        /** Checks if rows are linearly independent. */
        bool rows_linearly_independent();

        /** Checks if columns are linearly independent. */
        bool columns_linearly_independent();

        /*=========================================================
          Multiplication operators
          =========================================================*/

        /** Matrix multiplication operator with anonymous matrix. */
        virtual AnonymousMatrix<Scalar> operator*(AnonymousMatrix<Scalar> m) const;

        /** Matrix multiplication operator. */
        virtual AnonymousMatrix<Scalar> operator*(Matrix<Scalar> const& m);

        /** Assignment operator with matrix multiplication. */
        Matrix<Scalar>& operator*=(Matrix<Scalar> const& m);

        /** Assignment operator with anonymous matrix multiplication. */
        Matrix<Scalar>& operator*=(AnonymousMatrix<Scalar> m);

        /*=========================================================
          Assignments
          =========================================================*/

        /** Assignment operator with matrix. */
        Matrix<Scalar>& operator=(Matrix<Scalar> const& m);

        /** Assignment operator with values string. */
        Matrix<Scalar>& operator=(char const * values);

        /** Assignment operator with anonymous matrix. */
        Matrix<Scalar>& operator=(AnonymousMatrix<Scalar> m);

        /*=========================================================
          Retrieval and cast operators
          =========================================================*/

        /** Element retrieval with one index. */
        Scalar& operator() (int i) {
	    if ( rows == 1 || columns == 1 )
	        return values->at(i);
	    else
	        throw(NotAVectorException());
	}

        /** Element retrieval with one index (const). */
        Scalar const& operator() (int i) const {
	    if ( rows == 1 || columns == 1 )
	        return values->at(i);
	    else
	        throw(NotAVectorException());
	}

        /** Element retrieval with two idices. */
        Scalar& operator() (int r, int c) {
	    return values->at(r * columns + c);
	}

        /** Element retrieval with two indices (const). */
        Scalar const& operator() (int r, int c) const {
	    return values->at(r * columns + c);
	}

        /** Element retrieval with one index. */
        Scalar& at(int r, int c);

        /** Element retrieval with one index (const). */
        Scalar const& at(int r, int c) const;

        /** Implicit cast to double. */
        operator Scalar();

    protected:

        /** Procedure for Gaussian elimination. */
        AnonymousMatrix<Scalar> gaussian_elimination();

        /** Retrieves matrix type (anonymous or regular). */
        MatrixType get_matrix_type(Matrix<Scalar> const& m) const;

        /** Cache information. */
        mutable bool lu_up_to_date, determinant_up_to_date, inverse_up_to_date;

        /** Storage class, a reference counted pointer to heap-allocated data. */
        class storage {

        public:

            /** Constructor, destructor. */
            storage(int size);
            storage(int size, Scalar value);
            storage(storage& origin);
            ~storage();

            /** Access operator. */
            Scalar & at(int pos);

            /** Pointer to data. */
            std::vector< Scalar> * contents;

            /** Reference count. */
            int counter;

        };

        /** Pointer to heap-allocated storage (for implementing anonymous matrices in an efficient way). */
        storage* values;

        /** Number of rows. */
        int rows;

        /** Number of columns. */
        int columns;

        /** Determinant. */
        mutable Scalar det;

    };

    /** Represents a matrix for temporary use. */
    tempalte <typename Scalar>
    class AnonymousMatrix : public Matrix<Scalar> {

    public:

        /** Constructor and copy constructors. */
        AnonymousMatrix(int r, int c);
        AnonymousMatrix(const AnonymousMatrix<Scalar>& m);
        AnonymousMatrix(const Matrix<Scalar>& m);

        /** Multiplication operator. */
        AnonymousMatrix operator*(Matrix<Scalar> const& m);
    };

    /** Auxiliary function, number comparison with tolerance. */
    bool tol_equal(Scalar n, Scalar m, Scalar tol);
}

#endif
