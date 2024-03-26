#include "dot1d_conn_alea.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

void cree_dot1d_conn_alea(Mdl_t * mdl, uint c)
{
	uint X=mdl->Y[c-1], Y=mdl->Y[c];
	mdl->inst_POIDS        [c] = (CONNECTIONS+1)*mdl->Y[c];
	mdl->inst_VARS         [c] = mdl->Y[c];
	mdl->inst_LOCDS        [c] = mdl->Y[c];
	mdl->inst_SORTIES      [c] = mdl->Y[c];
	mdl->inst_DEPART_SORTIE[c] = mdl->Y[c] - mdl->Y[c];
	//
	mdl->p[c] = alloc<float>(mdl->inst_POIDS[c]);

	FOR(0, y, Y) {
		FOR(0, x, CONNECTIONS+1) {
			mdl->p[c][y*(CONNECTIONS+1)+x] = (2*rnd()-1) * sqrtf(/*10.0*/ 6.0 / (X+Y));
		}
	}
};

void plume_dot1d_conn_alea(Mdl_t * mdl, uint c)
{
	printf("POIDS dot1d_conn_alea: \n");
	uint X=mdl->Y[c-1], Y=mdl->Y[c];
	FOR(0, y, Y) {
		printf("y=%i : ", y);
		FOR(0, x, CONNECTIONS) {
			printf("%+f,", mdl->p[c][y*(CONNECTIONS+1)+x]);
		}
		printf(" biais=%+f\n", mdl->p[c][y*(CONNECTIONS+1)+CONNECTIONS+1-1]);
	}
};

void intel_dot1d_conn_alea(
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd)
{
/*#pragma omp parallel
#pragma omp for*/
	FOR(0, t, T) {
		FOR(0, _y, Y) {
			float s = p[_y*(CONNECTIONS+1)+(CONNECTIONS+1-1)];
			FOR(0, k, CONNECTIONS) {
				float __x = x[(0+t)*X_vars+DEPART_x+POINTEUR(_y,k,X)];
				float __p = p[_y*(CONNECTIONS+1)+k];
				s += __x * __p;
			}
			float a = dot1d_conn_alea_ACTIV(dot1d_conn_alea_ACTIVATION, s);
			y[(0+t)*Y+_y]    = a;
			locd[(0+t)*Y+_y] = dot1d_conn_alea_dACTIV(dot1d_conn_alea_ACTIVATION, s, a);
		}
	}
}

void d_intel_dot1d_conn_alea(
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
//#pragma omp parallel
//#pragma omp for
/*	FOR(0, t, T) {
		FOR(0, _y, Y) {
			float _locd = locd[(depart+t)*Y+_y] * dy[(depart+t)*Y+_y];
			dp[_y*(X+1)+(X+1-1)] += _locd;
			FOR(0, k, X) {
				//s += x[t*X+k] * p[y*(X+1)+k];
				dx[(depart+t)*X+k]      += _locd * p[_y*(X+1)+k];
				dp[_y*(X+1)+k] += _locd * x[(depart+t)*X+k];
			}
		}
	}*/

	//dx = (p @ ((y-_y)*dtanh(x@p)).T).T
/*#pragma omp parallel
#pragma omp for*/
	FOR(0, t, T) {
		FOR(0, _x, X) {
			//float _locd = locd[(depart+t)*Y+_y] * dy[(depart+t)*Y+_y];
			float s = 0;
			FOR(0, k, Y) {
				int lui_est_connectee = -1;	//lui est connecté par un poid, qui sera de `lui_est_connectee` addresse
				FOR(0, i, CONNECTIONS) {
					if (POINTEUR(k,i,X) == _x) {
						lui_est_connectee = (int)i;
						float __x = p[k*(CONNECTIONS+1)+lui_est_connectee];//x[(depart+t)*X+k];
						float __p = locd[(depart+t)*Y+k] * dy[(depart+t)*Y+k];//p[_y*(X+1)+k];
						s += __x * __p;
					}
				}
			}
			dx[(depart+t)*X_vars+DEPART_x+_x]  += s;
		}
	}

	//dp = x.T @ ((y-_y)*dtanh(x@p))
/*#pragma omp parallel
#pragma omp for*/
	FOR(0, _y, Y) {
		float dbiais = 0;
		FOR(0, _x, CONNECTIONS) {
			float s = 0;
			FOR(0, t, T) {
				float __x = locd[(depart+t)*Y+_y] * dy[(depart+t)*Y+_y];//x[(depart+t)*X+k];
				float __p = x[(depart+t)*X_vars+DEPART_x+POINTEUR(_y,_x,X)];//p[_y*(X+1)+k];
				s += __x * __p;
				if (_x == 0) {	//	Biais
					dbiais += __x;
				}
			}
			dp[_y*(CONNECTIONS+1)+_x] += s;
		}
		dp[_y*(CONNECTIONS+1)+(CONNECTIONS+1-1)] += dbiais;
	}
}

//	=========================================================

void f_dot1d_conn_alea(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1, uint _t_MODE, uint GRAINE) {
	uint X=mdl->Y[inst-1], Y=mdl->Y[inst];
	uint X_vars=mdl->inst_VARS[inst-1], Y_vars=mdl->inst_VARS[inst];
	uint depart = 0;//t0;
	uint T = (t1-t0);
	ASSERT(T == mdl->T);
	uint DEPART_x = mdl->inst_DEPART_SORTIE[inst-1];
	if (mode == 0) {
		intel_dot1d_conn_alea(
			_t_MODE, GRAINE,
			X_vars, Y_vars,
			X, Y,
			depart, T,
			DEPART_x,
			mdl->y[inst-1], mdl->y[inst],
			mdl->p[inst],
			mdl->l[inst]);
	} else if (mode == 1) {
		nvidia_dot1d_conn_alea_naive(
			_t_MODE, GRAINE,
			X_vars, Y_vars,
			X, Y,
			depart, T,
			DEPART_x,
			mdl->y__d[inst-1], mdl->y__d[inst],
			mdl->p__d[inst],
			mdl->l__d[inst]);
	} else if (mode == 2) {
		nvidia_dot1d_conn_alea_shared(
			_t_MODE, GRAINE,
			X_vars, Y_vars,
			X, Y,
			depart, T,
			DEPART_x,
			mdl->y__d[inst-1], mdl->y__d[inst],
			mdl->p__d[inst],
			mdl->l__d[inst]);
	} else if (mode == 3) {
		nvidia_dot1d_conn_alea_shared_2_16(
			_t_MODE, GRAINE,
			X_vars, Y_vars,
			X, Y,
			depart, T,
			DEPART_x,
			mdl->y__d[inst-1], mdl->y__d[inst],
			mdl->p__d[inst],
			mdl->l__d[inst]);
	} else {
		ERR("Pas de mode %i pour cuda f(x)", mode);
	}
}

//	----------------------------

void df_dot1d_conn_alea(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1, uint _t_MODE, uint GRAINE) {
	uint X=mdl->Y[inst-1], Y=mdl->Y[inst];
	uint X_vars=mdl->inst_VARS[inst-1], Y_vars=mdl->inst_VARS[inst];
	uint depart = 0;//t0;
	uint T = (t1-t0);
	ASSERT(T == mdl->T);
	uint DEPART_x = mdl->inst_DEPART_SORTIE[inst-1];
	if (mode == 0) {
		d_intel_dot1d_conn_alea(
			_t_MODE, GRAINE,
			X_vars, Y_vars,
			X, Y,
			depart, T,
			DEPART_x,
			mdl->y[inst-1], mdl->y[inst],
			mdl->p[inst],
			mdl->l[inst],
			mdl->dy[inst],
			mdl->dy[inst-1],
			mdl->dp[inst]);
	} else if (mode == 1) {
		d_nvidia_dot1d_conn_alea_naive(
			_t_MODE, GRAINE,
			X_vars, Y_vars,
			X, Y,
			depart, T,
			DEPART_x,
			mdl->y__d[inst-1], mdl->y__d[inst],
			mdl->p__d[inst],
			mdl->l__d[inst],
			mdl->dy__d[inst],
			mdl->dy__d[inst-1],
			mdl->dp__d[inst]);
	} else if (mode == 2) {
		d_nvidia_dot1d_conn_alea_shared(
			_t_MODE, GRAINE,
			X_vars, Y_vars,
			X, Y,
			depart, T,
			DEPART_x,
			mdl->y__d[inst-1], mdl->y__d[inst],
			mdl->p__d[inst],
			mdl->l__d[inst],
			mdl->dy__d[inst],
			mdl->dy__d[inst-1],
			mdl->dp__d[inst]);
	} else if (mode == 3) {
		d_nvidia_dot1d_conn_alea_shared_2_16(
			_t_MODE, GRAINE,
			X_vars, Y_vars,
			X, Y,
			depart, T,
			DEPART_x,
			mdl->y__d[inst-1], mdl->y__d[inst],
			mdl->p__d[inst],
			mdl->l__d[inst],
			mdl->dy__d[inst],
			mdl->dy__d[inst-1],
			mdl->dp__d[inst]);
	} else {
		ERR("Pas de mode %i pour cuda df(x)", mode);
	}
}