/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * myNeuralNetworkBasedOnPCA.c
 *
 * Code generation for function 'myNeuralNetworkBasedOnPCA'
 *
 */

/* Include files */
#include "rt_nonfinite.h"
#include "myNeuralNetworkBasedOnPCA.h"

/* Function Definitions */
void myNeuralNetworkBasedOnPCA(const double x1[12], double b_y1[25])
{
  int k;
  int xoffset;
  double z1[25];
  double b_z1[300];
  double c[300];
  static const double w[300] = { 4.1473234152417593, 0.24342967496605294,
    6.0516032919014364, 0.49044860323957362, -1.8301011483358915,
    16.656297194376759, 3.0375427139988931, -0.42278220313526982,
    -2.3719974394852437, -2.8256192166062304, 15.762613492696021,
    8.46408791456112, 12.126530399398897, -2.113696573298748,
    -2.5054198890554398, 12.927364562970496, 7.7525489810114854,
    0.52764129776408542, -1.1698311290385766, -1.990513682500255,
    14.590646757371378, 12.169964283694947, 3.3091342904467913,
    4.645916149197288, 3.3901609469960663, -8.904154821302777,
    -4.0506231515954925, -5.5141208526033161, -4.3968160057871062,
    -2.8414528524230542, -6.25641317392286, -8.7265253254453636,
    -6.3480541210007679, -1.4311079208164321, -0.4606485530152814,
    -1.3887503039948628, -4.1733153596431958, -4.2783918125922549,
    1.1898893019878911, 0.18214667975323595, 2.8086752056092621,
    6.0210734532043269, 4.8207594958984927, 2.9665000593240194,
    2.4408982351732833, 6.2910270165350157, 0.28119874945079965,
    6.1274762605248592, 8.6891174061841543, 17.483121639143839,
    4.3189393654490154, 0.8796527510544434, 1.5768411274300358,
    1.6638187494501961, 0.43539025887735755, -0.95199887040357012,
    6.1699729546970223, 3.1277623089625122, -0.42710523660019317,
    -1.1430705707290387, -5.2350239429684287, 8.5528978780413958,
    -0.60399573140315155, -0.055435978310816196, -0.88364468333522406,
    -5.2592364146510935, -2.5205927155939825, -0.25117708663243721,
    0.25166889427136185, 1.4521086871023761, 22.730490151041181,
    4.4814061490650037, -1.3545920173490942, 5.570727910923905,
    25.995725739339697, -3.0605366138685834, -2.0452518706269056,
    0.41128056493486087, -1.9501690399081355, -1.8651069809885161,
    4.1252843672221751, -11.290956703233213, -7.2314610836681865,
    -0.051102776880559267, 1.6356151350326609, 4.0947286686590463,
    -2.623890239933294, 4.5246308165090481, 1.1196212643055694,
    1.2849252664978614, -1.55884208368208, -4.7914171051846122,
    -2.0537007093075288, 0.10366060731414879, 1.9504117719701251,
    3.4973105322643159, 1.130443755200905, -4.4068305655924309,
    -1.6491829120898804, 11.173555982498566, -0.44175376947803796,
    2.1091970047451887, -0.12864993822600357, -0.74065842382533709,
    -1.076631587993732, 0.76888214589210979, 9.892318834154608, 9.23480156690221,
    -1.4351640797067318, -1.682974419170727, 2.3196649027241079,
    -3.6243342526654687, 1.7822557917317978, 0.98530628324151071,
    -0.27355486648673455, -0.77846139122241, -2.0899803943072075,
    1.0100840237953734, 1.7111658528745066, 2.1225992268370621,
    -9.4321830923401535, -3.0574918344516533, -0.81078844302860886,
    -0.26661345339848264, 9.18763189161319, -6.2816627937393008,
    -0.99804844105908952, -2.7563127014108435, -1.9357803027647147,
    -0.39835934612990453, -2.7410357580773157, 14.513856885402296,
    5.9717843013158616, 0.46972414078018021, 1.9627377396755383,
    1.8438939680551436, 2.892963599175002, -0.15451535010432865,
    -0.2879949580345264, 0.74510005839535587, 1.5177903718009058,
    0.52013787803007949, -1.5862886701975998, -1.3141644027360437,
    -0.65735468329336244, 5.8651427462894183, 1.5755993152195196,
    -0.83442144520047534, -0.31939574040096735, -1.0242865214665231,
    5.8349081130705489, -1.8185032463944113, 2.1883688712448763,
    0.60447750476927453, -2.2832713833932283, 3.2536647599596451,
    7.241149889029165, -2.3338231046958033, -0.89463139684904114,
    0.91325129128507887, -3.0140281149430086, 7.0483133797623188,
    -0.94120712085752223, 0.33912204702924592, 0.71314326948824192,
    -1.1425446374513619, 0.09519994159254884, 0.89213668790127487,
    0.64734929031960631, 0.0011719032937266127, -2.2771101265859106,
    0.38557074000678543, 0.72261588212905059, -1.0554152645083799,
    -7.9270486997459511, 1.2801879170474961, -3.1901026437329891,
    -0.43975593614545455, 0.82589696731149775, 1.3492535021961372,
    0.66477745083773088, 8.4259023144719745, -8.6142903875364567,
    0.34878429358466434, -0.81271917819118511, 1.2654709317124417,
    -2.0660324949587086, -1.2529063941492113, 0.27852046101699507,
    -0.35922545937073608, 1.7683296591993842, -0.29641174481576188,
    -0.69376570799988035, 0.020432717685482309, 1.177870463370724,
    -3.3754701257989463, -1.1368148213888627, -1.0079432552701653,
    -0.55496118906662772, 9.8586926013416019, 3.4573415217830217,
    1.2219121117416663, 0.45070993677809063, 0.55208706106543981,
    -0.64288836896807933, 1.2746766217958603, -2.767005236643862,
    3.2626105650362223, -0.17195803310897748, 0.94207189567142058,
    -1.0187509839219695, -3.0225902845034978, -2.5346777951690718,
    -0.20556327324725718, 0.11650096276338213, 2.7428804301079683,
    1.6780716934208779, -1.8407026360369207, -1.5309772983317287,
    1.0181857378688157, -2.4101817457223333, -1.6670254680302323,
    -0.10601612494523889, -0.021487421249293072, 16.540780737734877,
    -3.8601802072447069, -0.793470389931148, 0.018733338462491422,
    -0.58325522592937962, 0.52983333707470548, 1.0351869540978105,
    -3.2789704346740662, 2.3033804045333097, 0.066618752034369247,
    -0.88339702520397911, -0.43803443617573629, 14.200789073179427,
    -0.87392874542930021, 0.52893363671935534, 0.030693221451037339,
    0.98336972676474865, -0.027478072754647559, -0.56456173555666578,
    0.1606054747697645, 0.70121174538483932, -1.2853747298093208,
    -0.025052328128560925, -0.67390153864962388, -1.6248002927226186,
    0.920344150881854, -4.7488857050576758, -1.1001532831263068,
    0.93568902897477668, -0.5658206597471046, 0.54787893539058719,
    4.9646977145493736, 0.80182466646271866, 0.68201164815500692,
    0.50294951184681957, 0.52227074971899035, -2.1382816299695873,
    -4.063601082709134, 0.28720450925940305, -0.51466057838657164,
    -0.41120335447761147, -1.3425513717984174, -0.036170366184601656,
    0.5382168157126378, 0.0049429903213067524, -0.08364933320410925,
    -3.5048044020817972, 0.950069145110801, 0.558433634605654,
    1.5070838493486483, 3.3691668981046257, 0.24903886081726717,
    0.92042235901824665, -0.012859911288359227, -0.079152200368389547,
    0.015523403885730688, -1.55466671391372, -0.014319754221095826,
    -2.8266769038799042, 0.23963361502580863, 0.8649992353113003,
    1.1376447752073571, 9.5515636691749126, 1.5414356266275246,
    -0.75469112078227363, -0.44924448334221195, -1.4359506082232647,
    -0.30154516797425746, 0.57253841712174458, -0.33797375625465031,
    -0.11570059512104174, -4.30600096320408, -2.0912434862393905,
    0.89547359187357212, 2.038609107268984, 7.1471567890253969 };

  int j;
  boolean_T nanInd;
  double c_z1;
  boolean_T b[25];
  boolean_T exitg1;

  /* MYNEURALNETWORKBASEDONPCA neural network simulation function. */
  /*  */
  /*  Generated by Neural Network Toolbox function genFunction, 29-Oct-2017 16:24:48. */
  /*   */
  /*  [y1] = myNeuralNetworkBasedOnPCA(x1) takes these arguments: */
  /*    x = 12xQ matrix, input #1 */
  /*  and returns: */
  /*    y = 25xQ matrix, output #1 */
  /*  where Q is the number of samples. */
  /*  ===== NEURAL NETWORK CONSTANTS ===== */
  /*  Layer 1 */
  /*  ===== SIMULATION ======== */
  /*  Input 1 */
  /*  no processing */
  /*  Layer 1 */
  /*  ===== MODULE FUNCTIONS ======== */
  /*  Negative Distance Weight Function */
  for (k = 0; k < 12; k++) {
    for (xoffset = 0; xoffset < 25; xoffset++) {
      c[xoffset + 25 * k] = w[xoffset + 25 * k] - x1[k];
    }
  }

  for (k = 0; k < 300; k++) {
    b_z1[k] = c[k] * c[k];
  }

  memcpy(&z1[0], &b_z1[0], 25U * sizeof(double));
  for (k = 0; k < 11; k++) {
    xoffset = (k + 1) * 25;
    for (j = 0; j < 25; j++) {
      z1[j] += b_z1[xoffset + j];
    }
  }

  /*  Competitive Transfer Function */
  for (k = 0; k < 25; k++) {
    c_z1 = -sqrt(z1[k]);
    b[k] = rtIsNaN(c_z1);
    z1[k] = c_z1;
  }

  nanInd = false;
  k = 0;
  exitg1 = false;
  while ((!exitg1) && (k < 25)) {
    if (b[k]) {
      nanInd = true;
      exitg1 = true;
    } else {
      k++;
    }
  }

  memset(&b_y1[0], 0, 25U * sizeof(double));
  xoffset = 1;
  c_z1 = z1[0];
  j = 0;
  if (rtIsNaN(z1[0])) {
    k = 2;
    exitg1 = false;
    while ((!exitg1) && (k < 26)) {
      xoffset = k;
      if (!rtIsNaN(z1[k - 1])) {
        c_z1 = z1[k - 1];
        j = k - 1;
        exitg1 = true;
      } else {
        k++;
      }
    }
  }

  if (xoffset < 25) {
    while (xoffset + 1 < 26) {
      if (z1[xoffset] > c_z1) {
        c_z1 = z1[xoffset];
        j = xoffset;
      }

      xoffset++;
    }
  }

  b_y1[j] = 1.0;
  xoffset = 0;
  for (j = 0; j < 1; j++) {
    if (nanInd) {
      xoffset++;
    }
  }

  if (0 <= xoffset - 1) {
    for (xoffset = 0; xoffset < 25; xoffset++) {
      b_y1[xoffset] = rtNaN;
    }
  }

  /*  Output 1 */
}

/* End of code generation (myNeuralNetworkBasedOnPCA.c) */
