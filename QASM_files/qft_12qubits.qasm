OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
creg c[12];

h q[0];
cu1(1.5707963267948966) q[1],q[0];
cu1(0.7853981633974483) q[2],q[0];
cu1(0.39269908169872414) q[3],q[0];
cu1(0.19634954084936207) q[4],q[0];
cu1(0.09817477042468103) q[5],q[0];
cu1(0.04908738521234052) q[6],q[0];
cu1(0.02454369260617026) q[7],q[0];
cu1(0.01227184630308513) q[8],q[0];
cu1(0.006135923151542565) q[9],q[0];
cu1(0.0030679615757712823) q[10],q[0];
cu1(0.0015339807878856412) q[11],q[0];
h q[1];
cu1(1.5707963267948966) q[2],q[1];
cu1(0.7853981633974483) q[3],q[1];
cu1(0.39269908169872414) q[4],q[1];
cu1(0.19634954084936207) q[5],q[1];
cu1(0.09817477042468103) q[6],q[1];
cu1(0.04908738521234052) q[7],q[1];
cu1(0.02454369260617026) q[8],q[1];
cu1(0.01227184630308513) q[9],q[1];
cu1(0.006135923151542565) q[10],q[1];
cu1(0.0030679615757712823) q[11],q[1];
h q[2];
cu1(1.5707963267948966) q[3],q[2];
cu1(0.7853981633974483) q[4],q[2];
cu1(0.39269908169872414) q[5],q[2];
cu1(0.19634954084936207) q[6],q[2];
cu1(0.09817477042468103) q[7],q[2];
cu1(0.04908738521234052) q[8],q[2];
cu1(0.02454369260617026) q[9],q[2];
cu1(0.01227184630308513) q[10],q[2];
cu1(0.006135923151542565) q[11],q[2];
h q[3];
cu1(1.5707963267948966) q[4],q[3];
cu1(0.7853981633974483) q[5],q[3];
cu1(0.39269908169872414) q[6],q[3];
cu1(0.19634954084936207) q[7],q[3];
cu1(0.09817477042468103) q[8],q[3];
cu1(0.04908738521234052) q[9],q[3];
cu1(0.02454369260617026) q[10],q[3];
cu1(0.01227184630308513) q[11],q[3];
h q[4];
cu1(1.5707963267948966) q[5],q[4];
cu1(0.7853981633974483) q[6],q[4];
cu1(0.39269908169872414) q[7],q[4];
cu1(0.19634954084936207) q[8],q[4];
cu1(0.09817477042468103) q[9],q[4];
cu1(0.04908738521234052) q[10],q[4];
cu1(0.02454369260617026) q[11],q[4];
h q[5];
cu1(1.5707963267948966) q[6],q[5];
cu1(0.7853981633974483) q[7],q[5];
cu1(0.39269908169872414) q[8],q[5];
cu1(0.19634954084936207) q[9],q[5];
cu1(0.09817477042468103) q[10],q[5];
cu1(0.04908738521234052) q[11],q[5];
h q[6];
cu1(1.5707963267948966) q[7],q[6];
cu1(0.7853981633974483) q[8],q[6];
cu1(0.39269908169872414) q[9],q[6];
cu1(0.19634954084936207) q[10],q[6];
cu1(0.09817477042468103) q[11],q[6];
h q[7];
cu1(1.5707963267948966) q[8],q[7];
cu1(0.7853981633974483) q[9],q[7];
cu1(0.39269908169872414) q[10],q[7];
cu1(0.19634954084936207) q[11],q[7];
h q[8];
cu1(1.5707963267948966) q[9],q[8];
cu1(0.7853981633974483) q[10],q[8];
cu1(0.39269908169872414) q[11],q[8];
h q[9];
cu1(1.5707963267948966) q[10],q[9];
cu1(0.7853981633974483) q[11],q[9];
h q[10];
cu1(1.5707963267948966) q[11],q[10];
h q[11];
