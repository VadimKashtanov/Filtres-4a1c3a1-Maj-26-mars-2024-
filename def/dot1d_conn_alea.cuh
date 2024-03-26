#pragma once

#include "insts.cuh"

#define CONNECTIONS 16

#define POINTEUR(y,i,X) ((PSEUDO_ALEA((PSEUDO_ALEA(y))+i)) % X)

#define TANH 0
#define LOGISTIC 1
#define BINAIRE 2

#define dot1d_conn_alea_ACTIVATION /*BINAIRE*/LOGISTIC

#define logistique_f(s)    (1.f/(1.f + expf(-s)))
#define logistique_df(s,a) (a * (1.0-a))

#define dot1d_conn_alea_ACTIV(mode, s) logistique_f(s)
#define dot1d_conn_alea_dACTIV(mode, s,a) logistique_df(s,a)

#include "mdl.cuh"

void cree_dot1d_conn_alea(Mdl_t * mdl, uint inst);
void plume_dot1d_conn_alea(Mdl_t * mdl, uint c);

//	============================================

void intel_dot1d_conn_alea(
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd);

void nvidia_dot1d_conn_alea_naive(
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd);

void nvidia_dot1d_conn_alea_shared(
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd);

void nvidia_dot1d_conn_alea_shared_2_16(
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd);

void f_dot1d_conn_alea(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1, uint _t_MODE, uint GRAINE);

//	============================================

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
	float * dp);

void d_nvidia_dot1d_conn_alea_naive(
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
	float * dp);

void d_nvidia_dot1d_conn_alea_shared(
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
	float * dp);

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
	float * dp);

void df_dot1d_conn_alea(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1, uint _t_MODE, uint GRAINE);