abstract type AbstractMethod end
abstract type ExplicitMethod <: AbstractMethod end
abstract type EmbeddedMethod <: AbstractMethod end


# ******************************************************************************
# Explicit methods
# ******************************************************************************
# ------------------------------------------------------------------------------
# Kutta's third-order method
#
# https://en.wikipedia.org/w/index.php?title=List_of_Runge%E2%80%93Kutta_methods#Kutta's_third-order_method
# ------------------------------------------------------------------------------
struct RK3 <: ExplicitMethod end


function tableau(::RK3)
    as = [0    0    0;
          1/2  0    0;
          -1   2    0]
    bs = [1/6, 2/3, 1/6]
    cs = [0,   1/2, 1]
    return as, bs, cs
end


# ------------------------------------------------------------------------------
# Third-order Strong Stability Preserving Runge-Kutta (SSPRK3) method
#
# https://en.wikipedia.org/w/index.php?title=List_of_Runge%E2%80%93Kutta_methods#Third-order_Strong_Stability_Preserving_Runge-Kutta_(SSPRK3)
struct SSPRK3 <: ExplicitMethod end


function tableau(::SSPRK3)
    as = [0    0    0;
          1    0    0;
          1/4  1/4  0]
    bs = [1/6, 1/6, 2/3]
    cs = [0,   1,   1/2]
    return as, bs, cs
end


# ------------------------------------------------------------------------------
# Classic fourth-order method
#
# https://en.wikipedia.org/w/index.php?title=List_of_Runge%E2%80%93Kutta_methods#Classic_fourth-order_method
# ------------------------------------------------------------------------------
struct RK4 <: ExplicitMethod end


function tableau(::RK4)
    as = [0    0    0    0;
          1/2  0    0    0;
          0    1/2  0    0;
          0    0    1    0]
    bs = [1/6, 1/3, 1/3, 1/6]
    cs = [0,   1/2, 1/2, 1]
    return as, bs, cs
end


# ------------------------------------------------------------------------------
# Explicit Tsitouras method
#
# Ch. Tsitouras, "Runge-Kutta Pairs of Order 5(4) Satisfying Only the First
# Column Simplifying Assumption", Comput. Math. with Appl., 62, 770 (2011)
# ------------------------------------------------------------------------------
struct Tsit5 <: ExplicitMethod end

function tableau(::Tsit5)
    a21 = 0.161

    a31 = -0.008480655492356989
    a32 = 0.335480655492357

    a41 = 2.8971530571054935
    a42 = -6.359448489975075
    a43 = 4.3622954328695815

    a51 = 5.325864828439257
    a52 = -11.748883564062828
    a53 = 7.4955393428898365
    a54 = -0.09249506636175525

    a61 = 5.86145544294642
    a62 = -12.92096931784711
    a63 = 8.159367898576159
    a64 = -0.071584973281401
    a65 = -0.028269050394068383

    as = [0.0  0.0  0.0  0.0  0.0  0.0;
          a21  0.0  0.0  0.0  0.0  0.0;
          a31  a32  0.0  0.0  0.0  0.0;
          a41  a42  a43  0.0  0.0  0.0;
          a51  a52  a53  a54  0.0  0.0;
          a61  a62  a63  a64  a65  0.0]

    bs = [0.09646076681806523,
          0.01,
          0.4798896504144996,
          1.379008574103742,
          -3.290069515436081,
          2.324710524099774]

    cs = [0.0, 0.161, 0.327, 0.9, 0.9800255409045097, 1.0]
    return as, bs, cs
end


# ******************************************************************************
# Embedded methods
# ******************************************************************************
# ------------------------------------------------------------------------------
# Runge–Kutta–Fehlberg method
#
# https://en.wikipedia.org/w/index.php?title=List_of_Runge%E2%80%93Kutta_methods#Fehlberg
# ------------------------------------------------------------------------------
struct RK45 <: EmbeddedMethod end


function tableau(::RK45)
    a21 = 1/4

    a31 = 3/32
    a32 = 9/32

    a41 = 1932/2197
    a42 = -7200/2197
    a43 = 7296/2197

    a51 = 439/216
    a52 = -8
    a53 = 3680/513
    a54 = -845/4104

    a61 = -8/27
    a62 = 2
    a63 = -3544/2565
    a64 = 1859/4104
    a65 = -11/40

    as = [0.0  0.0  0.0  0.0  0.0  0.0;
          a21  0.0  0.0  0.0  0.0  0.0;
          a31  a32  0.0  0.0  0.0  0.0;
          a41  a42  a43  0.0  0.0  0.0;
          a51  a52  a53  a54  0.0  0.0;
          a61  a62  a63  a64  a65  0.0]
    bs = [16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55]
    cs = [0, 1/4, 3/8, 12/13, 1, 1/2]
    bhats = [25/216, 0, 1408/2565, 2197/4104, -1/5, 0]

    return as, bs, cs, bhats
end


# ------------------------------------------------------------------------------
# Dormand–Prince
#
# https://en.wikipedia.org/w/index.php?title=List_of_Runge%E2%80%93Kutta_methods#Dormand%E2%80%93Prince
# ------------------------------------------------------------------------------
struct DP5 <: EmbeddedMethod end


function tableau(::DP5)
    a21 = 1/5

    a31 = 3/40
    a32 = 9/40

    a41 = 44/45
    a42 = -56/15
    a43 = 32/9

    a51 = 19372/6561
    a52 = -25360/2187
    a53 = 64448/6561
    a54 = -212/729

    a61 = 9017/3168
    a62 = -355/33
    a63 = 46732/5247
    a64 = 49/176
    a65 = -5103/18656

    a71 = 35/384
    a72 = 0
    a73 = 500/1113
    a74 = 125/192
    a75 = -2187/6784
    a76 = 11/84

    as = [0.0  0.0  0.0  0.0  0.0  0.0  0.0;
          a21  0.0  0.0  0.0  0.0  0.0  0.0;
          a31  a32  0.0  0.0  0.0  0.0  0.0;
          a41  a42  a43  0.0  0.0  0.0  0.0;
          a51  a52  a53  a54  0.0  0.0  0.0;
          a61  a62  a63  a64  a65  0.0  0.0
          a71  a72  a73  a74  a75  a76  0.0]
    bs = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    cs = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
    bhats = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]

    return as, bs, cs, bhats
end


# ------------------------------------------------------------------------------
# Embedded Tsitouras method
#
# Ch. Tsitouras, "Runge-Kutta Pairs of Order 5(4) Satisfying Only the First
# Column Simplifying Assumption", Comput. Math. with Appl., 62, 770 (2011)
# ------------------------------------------------------------------------------
struct ATsit5 <: EmbeddedMethod end


function tableau(::ATsit5)
    as, bs, cs = tableau(Tsit5())

    bhats = [0.00178001105222577714,
             0.0008164344596567469,
             -0.007880878010261995,
             0.1447110071732629,
             -0.5823571654525552,
             0.45808210592918697]

             0.001780011052226
             0.00816434459657
             -0.007880878010262
             0.144711007173263
             -0.582357165452555
             0.458082105929187

    return as, bs, cs, bhats
end
