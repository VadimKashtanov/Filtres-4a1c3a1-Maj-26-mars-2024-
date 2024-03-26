#include "S.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

#define pseudo_alea_d_une_grain(i) ((float)((121+(i%1234))*31 % 1001 ) / 1001.0)

//	===============================================================

static __global__ void kerd_nvidia_score_somme_USD_NORME(
	uint _t_MODE, uint GRAINE,
	float * y, uint depart, uint T,
	float * score, float * _PRIXS)
{
	//uint t = threadIdx.x + blockIdx.x + blockDim.x;
	float usd = 0.01;
	float delta;
	float max_delta, min_delta;
	FOR(0, t, T) {
		uint cuda_depart_plus_t = t_MODE_GENERALE(_t_MODE, GRAINE, depart, DEPART, FIN, t);

		float s = 0;
		float _y = y[(0 + t)*P + 0];
		float alea = 2*pseudo_alea_d_une_grain(t + ((uint)_y % 10001))-1;
		_y += alea * SCORE_Y_COEF_BRUIT;
		//
		delta = usd * _y * (_PRIXS[cuda_depart_plus_t+1] / _PRIXS[cuda_depart_plus_t] -1);
		usd += delta;
		if (t == 0) {
			max_delta = fabs(delta);
			min_delta = fabs(delta);
		}
		//
		if (usd < 0) usd = 0;
		if (delta > max_delta) max_delta = fabs(delta);
		if (delta < min_delta) min_delta = fabs(delta);
	}
	usd = 0.01;
	FOR(0, t, T) {
		uint cuda_depart_plus_t = t_MODE_GENERALE(_t_MODE, GRAINE, depart, DEPART, FIN, t);

		float s = 0;
		//FOR(0, p, P) {
		uint p = 0;
			float _y = y[(0 + t)*P + p];
			float alea = 2*pseudo_alea_d_une_grain(t + ((uint)_y % 10001))-1;
			_y += alea * SCORE_Y_COEF_BRUIT;
			//
			float delta = usd * _y * (_PRIXS[cuda_depart_plus_t+1] / _PRIXS[cuda_depart_plus_t] -1);
			usd += delta;
			//
			float importance = (/*usd*/fabs(delta) - min_delta)/(max_delta-min_delta);
			//
			s += importance*(P-p)*cuda_SCORE(
				_y, _PRIXS[cuda_depart_plus_t+p+1], _PRIXS[cuda_depart_plus_t], alea * SCORE_Y_COEF_BRUIT
			);
		//}
		atomicAdd(score, s);
	}
};

static __global__ void kerd_nvidia_score_somme(
	uint _t_MODE, uint GRAINE,
	float * y, uint depart, uint T,
	float * score, float * _PRIXS)
{
	uint t = threadIdx.x + blockIdx.x + blockDim.x;
	if (t < T) {
		uint cuda_depart_plus_t = t_MODE_GENERALE(_t_MODE, GRAINE, depart, DEPART, FIN, t);

		float s = 0;
		FOR(0, p, P) {
			float _y = y[(0 + t)*P + p];
			float alea = 2*pseudo_alea_d_une_grain(t + ((uint)_y % 10001))-1;
			_y += alea * SCORE_Y_COEF_BRUIT;
			s += (P-p)*cuda_SCORE(
				_y, _PRIXS[/*depart+t*/cuda_depart_plus_t+p+1], _PRIXS[/*depart+t*/cuda_depart_plus_t], alea * SCORE_Y_COEF_BRUIT
			);
		}
		atomicAdd(score, s);
	}
};

float nvidia_somme_score(float * y, uint depart, uint T, uint _t_MODE, uint GRAINE)
{
	float * somme_score__d = cudalloc<float>(1);
	CONTROLE_CUDA(cudaMemset(somme_score__d, 0, sizeof(float)*1));
	if (SCORE_AVEC_USD_NORME == false) {
		kerd_nvidia_score_somme<<<dim3(KERD(T,1)),dim3(1)>>>(
			_t_MODE, GRAINE,
			y, depart, T,
			somme_score__d, prixs__d
		);
	} else {
		kerd_nvidia_score_somme_USD_NORME<<<dim3(1),dim3(1)>>>(
			_t_MODE, GRAINE,
			y, depart, T,
			somme_score__d, prixs__d
		);
	}
	ATTENDRE_CUDA();
	float somme_score;
	CONTROLE_CUDA(cudaMemcpy(&somme_score, somme_score__d, sizeof(float)*1, cudaMemcpyDeviceToHost));
	CONTROLE_CUDA(cudaFree(somme_score__d));
	return somme_score;
};

float  nvidia_score_finale(float somme, uint T, uint _t_MODE, uint GRAINE) {
	return APRES_SCORE(somme / (float)(P * T));
};

/*	Regularisation L2
	Pas oublier le Attention mechanisme
*/

//	===============================================================

float d_nvidia_score_finale(float somme, uint T, uint _t_MODE, uint GRAINE) {
	return dAPRES_SCORE(somme / (float)(P * T)) / (float)(P * T);
};

//	===============================================================

static __global__ void kerd_nvidia_score_dpowf_USD_NORME(
	uint _t_MODE, uint GRAINE,
	float _dy, float * y, float * dy,
	uint depart, uint T,
	float * _PRIXS)
{
	float usd = 0.01;
	float delta;
	float max_delta, min_delta;
	FOR(0, t, T) {
		uint cuda_depart_plus_t = t_MODE_GENERALE(_t_MODE, GRAINE, depart, DEPART, FIN, t);

		float s = 0;
		float _y = y[(0 + t)*P + 0];
		float alea = 2*pseudo_alea_d_une_grain(t + ((uint)_y % 10001))-1;
		_y += alea * SCORE_Y_COEF_BRUIT;
		//
		delta = usd * _y * (_PRIXS[cuda_depart_plus_t+1] / _PRIXS[cuda_depart_plus_t] -1);
		usd += delta;
		if (t == 0) {
			max_delta = fabs(delta);
			min_delta = fabs(delta);
		}
		//
		if (usd < 0) usd = 0;
		if (delta > max_delta) max_delta = fabs(delta);
		if (delta < min_delta) min_delta = fabs(delta);
	}
	usd = 0.01;
	FOR(0, _t, T) {
		uint cuda_depart_plus_t = t_MODE_GENERALE(_t_MODE, GRAINE, depart, DEPART, FIN, _t);
		//FOR(0, p, P) {
		uint p = 0;
		float _y = y[(0+_t)*P+p];
		float alea = 2*pseudo_alea_d_une_grain(_t + ((uint)_y % 10001))-1;
		//
		_y += alea*SCORE_Y_COEF_BRUIT;
		//
		float delta = usd * _y * (_PRIXS[cuda_depart_plus_t+1] / _PRIXS[cuda_depart_plus_t] -1);
		usd += delta;
		//
		float importance = (/*usd*/fabs(delta) - min_delta)/(max_delta-min_delta);
		//
		dy[(0+_t)*P+p] = _dy * importance * (P-p)*cuda_dSCORE(
			_y, _PRIXS[cuda_depart_plus_t+p+1], _PRIXS[cuda_depart_plus_t], alea*SCORE_Y_COEF_BRUIT
		);
		//}
	}
};

static __global__ void kerd_nvidia_score_dpowf(
	uint _t_MODE, uint GRAINE,
	float _dy, float * y, float * dy,
	uint depart, uint T,
	float * _PRIXS)
{
	uint _t = threadIdx.x + blockIdx.x * blockDim.x;

	if (_t < T) {
		uint cuda_depart_plus_t = t_MODE_GENERALE(_t_MODE, GRAINE, depart, DEPART, FIN, _t);
		FOR(0, p, P) {
			float _y = y[(0+_t)*P+p];
			float alea = 2*pseudo_alea_d_une_grain(_t + ((uint)_y % 10001))-1;
			dy[(0+_t)*P+p] = _dy * (P-p)*cuda_dSCORE(
				y[(0+_t)*P+p]+alea*SCORE_Y_COEF_BRUIT, _PRIXS[/*depart+_t*/cuda_depart_plus_t+p+1], _PRIXS[/*depart+_t*/cuda_depart_plus_t/*+p*/], alea*SCORE_Y_COEF_BRUIT
			);
		}
	}
};

void d_nvidia_somme_score(float d_score, float * y, float * dy, uint depart, uint T, uint _t_MODE, uint GRAINE) {
	if (SCORE_AVEC_USD_NORME == false) {
		kerd_nvidia_score_dpowf<<<dim3(KERD(T,1024)), dim3(1024)>>>(
			_t_MODE, GRAINE,
			d_score,
			y, dy,
			depart, T,
			prixs__d
		);
	} else if (SCORE_AVEC_USD_NORME == true) {
		kerd_nvidia_score_dpowf_USD_NORME<<<dim3(1), dim3(1)>>>(
			_t_MODE, GRAINE,
			d_score,
			y, dy,
			depart, T,
			prixs__d
		);
	}
	ATTENDRE_CUDA();
};