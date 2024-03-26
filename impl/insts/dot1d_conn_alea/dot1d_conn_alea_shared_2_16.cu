#include "dot1d_conn_alea.cuh"

/*	Difference :
	Au lieux de directement deriver avec que des atomicAdd le
__shared__ noyau, on fait la méthode que j'avais avant
ou on fait une autre opération pour calc dx et dp.

	Mathématiquement ca correspond a deriver y=X@P+B
en dX=p@dY.T
	dx = (p @ ((y-_y)*dtanh(x@p)).T).T
	dp = x.T @ ((y-_y)*dtanh(x@p))
*/

#define BLOQUE 16
#define BLOQUE_MAX 16

static __global__ void kerd_stricte_16__shared2(
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd)
{
	// <KERD(T, BLOQUE), KERD(Y,BLOQUE)>
	// <         BLOQUE,         BLOQUE>

	__shared__ float __partage__x[BLOQUE][BLOQUE];
	__shared__ float __partage__p[BLOQUE][BLOQUE];

	uint thx = threadIdx.x;
	uint thy = threadIdx.y;

	uint _y = thx + blockIdx.x * blockDim.x;
	uint _t = thy + blockIdx.y * blockDim.y;

	float s = 0;

	FOR(0, d, X/BLOQUE) {
		__partage__x[thy][thx] = x[(0+_t)*( X_vars ) + DEPART_x + POINTEUR(_y,d*BLOQUE + thx,X)];
		__partage__p[thy][thx] = p[_y*(CONNECTIONS+1) + d*BLOQUE + thy];
		__syncthreads();

	//#pragma unroll
		FOR(0, i, BLOQUE) s += __partage__x[thy][i] * __partage__p[i][thx];
		__syncthreads();
	};

#define __partage__b __partage__x[0]

	if (thy == 0) __partage__b[thx] = p[_y*(CONNECTIONS+1) + (CONNECTIONS+1-1)];
	__syncthreads();

	s = (s + __partage__b[thx]);
	float a = dot1d_conn_alea_ACTIV(ACTIVATION, s);
	   y[(0+_t)*Y + _y] = a;
	locd[(0+_t)*Y + _y] = dot1d_conn_alea_dACTIV(ACTIVATION, s,a);
};

void nvidia_dot1d_conn_alea_shared_2_16(
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd)
{
	if (T%BLOQUE!=0) ERR("ATTENTION T%%16 != 0 (T=%i)", T);
	if (X%BLOQUE==0 && Y%BLOQUE==0 && T%BLOQUE==0) {
		kerd_stricte_16__shared2<<<dim3(KERD(Y, BLOQUE), KERD(T, BLOQUE)), dim3(BLOQUE, BLOQUE)>>>(
			_t_MODE, GRAINE,
			X_vars, Y_vars,
			X, Y,
			depart, T,
			DEPART_x,
			x, y,
			p,
			locd);
		ATTENDRE_CUDA();
	} else {
		nvidia_dot1d_conn_alea_naive(
			_t_MODE, GRAINE,
			X_vars, Y_vars,
			X, Y,
			depart, T,
			DEPART_x,
			x, y,
			p,
			locd);
	}
}

static __global__ void kerd_stricte_16__shared2____dx(
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd,
	float * dy,
	float * dx,
	float * dp)
{

	//dx = (p @ ((y-_y)*dtanh(x@p)).T).T

	__shared__ float __partage__x[BLOQUE][BLOQUE];
	__shared__ float __partage__p[BLOQUE][BLOQUE];

	uint thx = threadIdx.x;
	uint thy = threadIdx.y;

	//uint _y = thx + blockIdx.x * blockDim.x;
	uint _x = thx + blockIdx.x * blockDim.x;
	uint _t = thy + blockIdx.y * blockDim.y;

	float s = 0;

	FOR(0, d, Y/BLOQUE) {
		int lui_est_connectee = -1;	//lui est connecté par un poid, qui sera de `lui_est_connectee` addresse
		FOR(0, i, CONNECTIONS) {
			if (POINTEUR((d*BLOQUE+thx),i,X) == _x) {
				lui_est_connectee = (int)i;
			}
		}
		if (lui_est_connectee != -1) {
			__partage__x[thy][thx] = locd[(0+_t)*Y+d*BLOQUE+thx] * dy[(0+_t)*Y+d*BLOQUE+thx];//x[(depart+_t)*( X ) + d*BLOQUE + thx];
			__partage__p[thy][thx] = p[(d*BLOQUE+thy)*(CONNECTIONS+1) + lui_est_connectee];//p[_y*(X+1) + d*BLOQUE + thy];
		}
		__syncthreads();

	//#pragma unroll
		FOR(0, i, BLOQUE) s += __partage__x[thy][i] * __partage__p[i][thx];
		__syncthreads();
	};

	dx[(0+_t)*X_vars+DEPART_x +_x]   = s;
	//printf("s=%f\n", s);
};


static __global__ void kerd_stricte_32__shared2____dp(
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd,
	float * dy,
	float * dx,
	float * dp)
{

	//dp = x.T @ ((y-_y)*dtanh(x@p))

	__shared__ float __partage__x[BLOQUE_MAX][BLOQUE_MAX];
	__shared__ float __partage__p[BLOQUE_MAX][BLOQUE_MAX];

	uint thx = threadIdx.x;
	uint thy = threadIdx.y;

	uint _x = thx + blockIdx.x * blockDim.x;
	uint _y = thy + blockIdx.y * blockDim.y;

	float s = 0;
	float biais = 0;

	uint d = blockIdx.z;
	//FOR(0, d, T/BLOQUE) {
		__partage__x[thy][thx] = locd[(0+d*BLOQUE_MAX+thx)*Y+_y] * dy[(0+d*BLOQUE_MAX+thx)*Y+_y];//x[(depart+_t)*( X ) + d*BLOQUE + thx];
		__partage__p[thy][thx] = x[(0+(d*BLOQUE_MAX+thy))*X_vars+DEPART_x +POINTEUR(_y,_x,X)];//p[_y*(X+1) + d*BLOQUE + thy];
		__syncthreads();

	#pragma unroll
		FOR(0, i, BLOQUE_MAX) {
			s += __partage__x[thy][i] * __partage__p[i][thx];
			if (_x == 0) biais += __partage__x[thy][i];
		}
		__syncthreads();
	//};

#define __partage__b __partage__x[0]

	//if (thy == 0) __partage__b[thx] = p[_y*(X+1) + (X+1-1)];
	if (_x == 0) atomicAdd(&dp[_y*(CONNECTIONS+1) + (CONNECTIONS+1-1)], biais);
	__syncthreads();

	atomicAdd(&dp[_y*(CONNECTIONS+1)+_x], s);
};

void d_nvidia_dot1d_conn_alea_shared_2_16(
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd,
	float * dy,
	float * dx,
	float * dp)
{
	if (T%(MAX2(BLOQUE_MAX,BLOQUE))!=0) ERR("ATTENTION T%%%i != 0 (T=%i)", T, (MAX2(BLOQUE_MAX,BLOQUE)));
	if (X%BLOQUE==0 && Y%BLOQUE==0 && T%BLOQUE==0) {
		kerd_stricte_16__shared2____dx<<<dim3(KERD(/*X*/CONNECTIONS, BLOQUE), KERD(T, BLOQUE)), dim3(BLOQUE, BLOQUE)>>>(
			_t_MODE, GRAINE,
			X_vars, Y_vars,
			X, Y,
			depart, T,
			DEPART_x,
			x, y,
			p,
			locd,
			dy,
			dx,
			dp);
		kerd_stricte_32__shared2____dp<<<dim3(KERD(/*X*/CONNECTIONS, BLOQUE_MAX), KERD(Y, BLOQUE_MAX), DIV(T,BLOQUE_MAX)), dim3(BLOQUE_MAX, BLOQUE_MAX,1)>>>(
			_t_MODE, GRAINE,
			X_vars, Y_vars,
			X, Y,
			depart, T,
			DEPART_x,
			x, y,
			p,
			locd,
			dy,
			dx,
			dp);
		ATTENDRE_CUDA();
	} else {
		d_nvidia_dot1d_conn_alea_naive(
			_t_MODE, GRAINE,
			X_vars, Y_vars,
			X, Y,
			depart, T,
			DEPART_x,
			x, y,
			p,
			locd,
			dy,
			dx,
			dp);
	}
}