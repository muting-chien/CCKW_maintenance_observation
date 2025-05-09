#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#define NP           33      /* number of pressure divisions */
#define BEGINP    17500.0      
#define DP         2500.0
#define R           287.0
#define CP         1004.0
#define KTOP(K)   ( BEGINP + DP * ((double) K - 0.5) )		/* for phi, u, v */
#define KTOP2(K)  ( BEGINP + DP * (double) K )			/* for T, S, mu*/
#define SIGN(A,B) ( (B>0.0) ? (A) : -(A) )

double
ddp(double *phi, int k)
{
  if (k == 0 || k == NP) 
     return 0.0;
  else
     return (phi[k+1] - phi[k]) / DP;
}


double 
pythag(double a, double b)
{
   return sqrt(a*a + b*b);
}

double
dot(double v1[NP], double v2[NP])  
{
  int k;
  double result = 0.0;

  for (k = 1; k <= NP; ++k) 
    result += v1[k] * v2[k];
  return result / (double) NP;
}

void
tri(double d[], double e[], double z[NP+1][NP+1])  /* z is a NP by NP matrix */
{
   int m, l, iter, i, k;
   double s, r, p, g, f, dd, c, b;
   for (l = 1; l <= NP; ++l) {
      iter = 0;
      do {
         for (m = l; m < NP; ++m) {
            dd = fabs(d[m]) + fabs(d[m+1]);
            if ( (fabs(e[m]) + dd) == dd ) {
               /* printf("skipping\n"); */
               break;
	    }
         }
         if (m != l) {
            if (++iter == 30) {
               fprintf(stderr, "too many iterations\n");
               exit(-1);
            }
            g = (d[l+1] - d[l]) / (2.0 * e[l]);
            r = pythag(g, 1.0);
            g = d[m] - d[l] + e[l] / (g + SIGN(r, g));
            s = c = 1.0;
            p = 0.0;
            for ( i = m-1; i >= l; --i) {
               f = s * e[i];
               b = c * e[i];
               e[i+1] = (r=pythag(f,g));
               if (fabs(r) == 0.0) {
                  d[i+1] -= p;
                  e[m] = 0.0;
                  break;
               }
	       /* printf("%f\n", r); */
               s = f/r;
               c = g/r;
               g = d[i+1] - p;
               r = (d[i] - g) * s + 2.0 * c * b;
               d[i+1] = g + (p=s*r);
               g = c * r - b;
               for (k = 1; k <= NP; ++k) {
                  f = z[k][i+1];
                  z[k][i+1] = s * z[k][i] + c * f;
                  z[k][i] = c * z[k][i] - s * f;
               }
            }
            if ( r == 0.0 && i) continue;
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
         }
      } while (m != l);
   }
}
 
int
main(int argc, char *argv[])
{
   int k, l;
   double p, t[NP+1], om[NP+1], stab, mu[NP+1];
   double diag[NP+1], offdiag[NP], z[NP+1][NP+1], mode[NP+1];
   double sum, phi0[NP+1], phi[NP+1], proj[NP+1], mag;
   double c[NP+1], c1, c2;  
   FILE *file;

   if (argc < 2) {
      fprintf(stderr, "mode temperature_file [t||h||o mode_number]\n");
      exit(-1);
   }

   /* reading stability from file and calculating mu (Haertel and Johnson 1998, Appendix) */
   if ((file = fopen(argv[1], "r")) == NULL) {
      fprintf(stderr, "cannot open file %s\n", argv[1]);
      exit(-1);
   }
   for (k = 0; k <= NP; ++k) {
      if (fscanf(file, "%lf %lf", &p, t+k) != 2 || p * 100.0 != KTOP2(k)) {
         fprintf(stderr, "temperature file error at pressure %f\n", p);
         exit(-1);
      }
   }
   fclose(file);
   for (k = 1; k < NP; ++k) {
      stab = -(t[k+1] - t[k-1]) / (2.0 * DP) + (R / CP) * t[k] / KTOP2(k);
      mu[k] = KTOP2(k) / (R * stab);
   }

   /* defining the matrix representation of the linear operator 
    * and finding the eigenvectors and eigenvalues */
   for (k = 1; k < NP; ++k)
      offdiag[k] = mu[k] / (DP * DP);
   diag[1] = -offdiag[1];
   for (k = 2; k < NP; ++k) 
      diag[k] = -offdiag[k-1] - offdiag[k];
   diag[NP] = -offdiag[NP-1];
   for (k = 1; k <= NP; ++k) {
         for (l = 1; l <= NP; ++l) {
            if (k == l)
               z[k][l] = 1.0;
	    else
               z[k][l] = 0.0;
	 }
   }
   tri(diag, offdiag, z);  
   for (k = 1; k <= NP; ++k) {
      for (l = 1; l <= NP; ++l) 
         z[k][l] *= sqrt((double) NP);
   }
   for (l = 1; l <= NP; ++l) 
      c[l] = 1.0 / sqrt(fabs(diag[l]));
   
   /* no input--list phase speed for each mode */
   if (argc == 2) {
      for (l = 1; l <= NP; ++l) 
         printf( "%2d %6.2f\n", l, c[l]);
   }

   /* geopotential structure for particular mode */
   else if (strcmp(argv[2], "m") == 0) {
	 if (argc < 4) {
		 fprintf(stderr, "usage: %s t_av m #\n", argv[0]);
		 exit(-1);
	 }
         l = atoi(argv[3]);
         for (k = 1; k <= NP; ++k)
            printf("%5.0f %f\n", KTOP(k), z[k][l]);
   }

   /* temperature input */
   else if (strcmp(argv[2], "t") == 0) {
      for (k = 0; k <= NP; ++k) {
         if (scanf("%lf %lf", &p, t + k) != 2 || p != KTOP2(k)) {
            fprintf(stderr, "can't parse input %f\n", KTOP(k));
            exit(-1);
         }
      }
      /* convert to geopotential */
      phi[1] = 0.0;
      for(k = 1; k < NP; ++k) 
         phi[k+1] = phi[k] - DP * (R * t[k] / KTOP2(k));
      for(k = 1, sum = 0.0; k <= NP; ++k)
        sum += phi[k];
      for(k = 1; k <= NP; ++k)
        phi[k] -= sum / (double) NP;
      /* no mode requested--print coefficient of each mode */
      if (argc == 3) {
         for (l = 1; l <= NP; ++l) {
            for (k = 1; k <= NP; ++k) 
               mode[k] = z[k][l];
            printf( "%2d %8.2f %6.3f\n", l, 1.0 / sqrt(fabs(diag[l])), dot(phi, mode));
         }
      }
      /* projection onto requested mode */
      else if (argc == 4) {
         l = atoi(argv[3]);
         for (k = 1; k <= NP; ++k)
            mode[k] = z[k][l];
         mag = dot(phi, mode);
         for (k = 1; k <= NP; ++k)
            proj[k] = mag * mode[k];
         for (k = 0; k <= NP; ++k) 
            printf("%5.0f %6.3f\n", KTOP2(k), -ddp(proj,k) * KTOP2(k) / R);
      }
      /* projection onto range of modes */
      else {
	 c1 = atof(argv[3]);
	 c2 = atof(argv[4]);
	 for (k = 1; k <= NP; ++k)
            proj[k] = 0.0;
	 for (l = 1; l <= NP; ++l) {
            if (c[l] < c1 || c[l] > c2) continue;
            for (k = 1; k <= NP; ++k)
               mode[k] = z[k][l];
            mag = dot(phi, mode);
            for (k = 1; k <= NP; ++k)
               proj[k] += mag * mode[k];
	 }
         for (k = 0; k <= NP; ++k) 
            printf("%5.0f %6.3f\n", KTOP2(k), -ddp(proj,k) * KTOP2(k) / R);
      }
   }

   /* geopotential/wind input */
   else if (strcmp(argv[2], "h") == 0) {
      for (k = 0; k <= NP; ++k) {
         if (scanf("%lf %lf", &p, phi0 + k) != 2 || p != KTOP2(k)) {
            fprintf(stderr, "can't parse input %f\n", KTOP(k));
            exit(-1);
         }
      }
      /* interpolate to different pressures */
      for (k = 1;  k <= NP; ++k)
         phi[k] = 0.5 * (phi0[k-1] + phi0[k]);
      /* no mode requested--print coefficient of each mode */
      if (argc == 3) {
         for (l = 1; l <= NP; ++l) {
            for (k = 1; k <= NP; ++k) 
               mode[k] = z[k][l];
            printf( "%2d %8.2f %6.3f\n", l, 1.0 / sqrt(fabs(diag[l])), dot(phi, mode));
         }
      }
      /* projection onto requested mode */
      else if (argc == 4) {
         l = atoi(argv[3]);
         for (k = 1; k <= NP; ++k)
            mode[k] = z[k][l];
         mag = dot(phi, mode);
         for (k = 1; k <= NP; ++k)
            proj[k] = mag * mode[k];
         for (k = 1; k <= NP; ++k) 
            printf("%5.0f %6.3f\n", KTOP(k), proj[k]);
      }
      /* projection onto range of modes */
      else {
	 c1 = atof(argv[3]);
	 c2 = atof(argv[4]);
	 for (k = 1; k <= NP; ++k)
            proj[k] = 0.0;
	 for (l = 1; l <= NP; ++l) {
            if (c[l] < c1 || c[l] > c2) continue;
            for (k = 1; k <= NP; ++k)
               mode[k] = z[k][l];
            mag = dot(phi, mode);
            for (k = 1; k <= NP; ++k)
               proj[k] += mag * mode[k];
	 }
         for (k = 1; k <= NP; ++k) 
            printf("%5.0f %6.3f\n", KTOP(k), proj[k]);
      }
   }

   /* omega (pressure velocity) input */
   else if (strcmp(argv[2], "o") == 0) {
      for (k = 0; k <= NP; ++k) {
         if (scanf("%lf %lf", &p, om + k) != 2 || p != KTOP2(k)) {
            fprintf(stderr, "can't parse input %f\n", KTOP(k));
            exit(-1);
         }
      }
      /* convert to divergence */
      for(k = 1; k <= NP; ++k) 
         phi[k] = om[k-1] - om[k];	  // doesn't matter what units are--will convert back to orig.
      // projection onto range of modes
      c1 = atof(argv[3]);
      c2 = atof(argv[4]);
      for (k = 1; k <= NP; ++k)
            proj[k] = 0.0;
      for (l = 1; l <= NP; ++l) {
         if (c[l] < c1 || c[l] > c2) continue;
         for (k = 1; k <= NP; ++k)
            mode[k] = z[k][l];
         mag = dot(phi, mode);
         for (k = 1; k <= NP; ++k)
            proj[k] += mag * mode[k];
      }
      om[NP] = 0.0;
      for (k = NP-1; k >= 0; --k) 
          om[k] = om[k+1] + proj[k+1];
      for (k = 0; k <= NP; ++k) 
         printf("%5.0f %6.3f\n", KTOP2(k), om[k]);
   }


   else {
	   fprintf(stderr, "unknown option: %s\n", argv[2]);
	   exit(-1);
   }
      
   exit(0);
}
