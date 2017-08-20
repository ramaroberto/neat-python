.. _activation-functions-label:

Overview of builtin activation functions
========================================

.. index:: ! activation function

Note that some of these :term:`functions <activation function>` are scaled differently from the canonical
versions you may be familiar with.  The intention of the scaling is to place
more of the functions' "interesting" behavior in the region :math:`\left[-1, 1\right] \times \left[-1, 1\right]`.
Some of these are more intended for :term:`CPPNs <CPPN>` (e.g., for :term:`HyperNEAT`) than for "direct" problem-solving,
as noted below; however, even those meant mainly for CPPNs can be of use elsewhere - `abs` and ``hat`` can both solve xor in one generation,
for instance (although note for the former that it is included in several others such as :ref:`multiparam_relu <multiparam-relu-description-label>`).

The :term:`multiparameter` functions below, and some of the others, are new; if users wish to try substituting them for previously-used activation functions, the following are suggested:

======== =======================================================
Old            New
======== =======================================================
abs             multiparam_relu, multiparam_relu_softplus, or weighted_lu
clamped      :ref:`clamped_tanh_step <clamped-tanh-step-label>`
cube           :ref:`multiparam_pow <multiparam-pow-label>`
gauss         :ref:`hat_gauss <hat-gauss-label>`
hat             :ref:`hat_gauss <hat-gauss-label>`
identity       multiparam_relu, multiparam_relu_softplus, or weighted_lu
inv             :ref:`multiparam_log_inv <multiparam-log-inv-label>`, if for a CPPN
log             :ref:`scaled_expanded_log <scaled-expanded-log-label>`, if for a CPPN
relu            multiparam_relu, multiparam_relu_softplus, or weighted_lu
sigmoid      :ref:`multiparam_sigmoid <multiparam-sigmoid-label>`
softplus      :ref:`multiparam_relu_softplus <multiparam-relu-softplus-label>`
square        :ref:`multiparam_pow <multiparam-pow-label>`
tanh           :ref:`clamped_tanh_step <clamped-tanh-step-label>`
======== =======================================================

The implementations of these functions can be found in the :py:mod:`activations` module.

General-use activation functions (single-parameter)
-----------------------------------------------------------------------

clamped
^^^^^^^^^

.. figure:: activation-clamped.png
   :scale: 100 %
   :alt: clamped linear function

log1p
^^^^^^

.. figure:: activation-log1p.png
    :scale: 100 %
    :alt: log(x+1) function with alterations for negative numbers

relu
^^^^

.. figure:: activation-relu.png
   :scale: 100 %
   :alt: rectified linear function (max(x,0))

.. _sigmoid-label:

sigmoid
^^^^^^^

.. figure:: activation-sigmoid.png
   :scale: 100 %
   :alt: sigmoid function

softplus
^^^^^^^^

.. figure:: activation-softplus.png
   :scale: 100 %
   :alt: soft-plus function (effectively a version of relu with a curve around 0)

step
^^^^

.. figure:: activation-step.png
    :scale: 100%
    :alt: step function: -1 below 0, 0 at exactly 0, 1 above 0

.. _tanh-label:

tanh
^^^^

.. figure:: activation-tanh.png
   :scale: 100 %
   :alt: hyperbolic tangent function

General-use activation functions (multiparameter)
---------------------------------------------------------------------

.. _clamped-tanh-step-label:

clamped_tanh_step
^^^^^^^^^^^^^^^^^^

.. figure:: activation-clamped_tanh_step.png
    :scale: 100 %
    :alt: Weighted combination of clamped, :ref:`tanh <tanh-label>`, and step functions.

multiparam_elu
^^^^^^^^^^^^^^^

.. figure:: activation-multiparam_elu.png
    :scale: 100 %
    :alt: Variable-scaling version of the exponential linear function (ELU)

.. figure:: activation-swap-multiparam_elu.png
    :scale: 100 %
    :alt: Variable-scaling version of the exponential linear function (ELU)

.. _multiparam-relu-description-label:

multiparam_relu
^^^^^^^^^^^^^^^

.. figure:: activation-multiparam_relu.png
    :scale: 100 %
    :alt: max(x, a*x), where a is an evolved parameter with a range from -1 to 1, inclusive. Acts like a weighted combination of abs, relu, and identity.

.. _multiparam-relu-softplus-label:

multiparam_relu_softplus
^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: activation-multiparam_relu_softplus.png
    :scale: 100 %
    :alt: A weighted combination of softplus, relu, abs, and identity.

.. figure:: activation-swap-multiparam_relu_softplus.png
    :scale: 100 %
    :alt: A weighted combination of softplus, relu, abs, and identity.

.. _multiparam-sigmoid-label:

multiparam_sigmoid
^^^^^^^^^^^^^^^^^^^

.. figure:: activation-multiparam_sigmoid.png
    :scale: 100 %
    :alt: A version of :ref:`clamped_tanh_step <clamped-tanh-step-label>` rescaled to match :ref:`sigmoid <sigmoid-label>` instead of :ref:`tanh <tanh-label>`.

multiparam_tanh_log1p
^^^^^^^^^^^^^^^^^^^^^

.. figure:: activation-multiparam_tanh_log1p.png
    :scale: 100 %
    :alt: A weighted combination of :ref:`clamped_tanh_step <clamped-tanh-step-label>` and scaled_log1p.

.. figure:: activation-swap-multiparam_tanh_log1p.png
    :scale: 100 %
    :alt: A weighted combination of :ref:`clamped_tanh_step <clamped-tanh-step-label>` and scaled_log1p.

scaled_log1p
^^^^^^^^^^^^^

.. figure:: activation-scaled_log1p.png
    :scale: 100 %
    :alt: A version of log1p with variable scaling (with partially-counterbalancing weights inside and outside the log1p function).

weighted_lu
^^^^^^^^^^^^

.. figure:: activation-weighted_lu.png
    :scale: 100 %
    :alt: A weighted combination of multiparam_relu and multiparam_elu.

.. figure:: activation-swap-weighted_lu.png
    :scale: 100 %
    :alt: A weighted combination of multiparam_relu and multiparam_elu.

CPPN-intended activation functions (single-parameter)
----------------------------------------------------------------------------

abs
^^^

.. figure:: activation-abs.png
   :scale: 100 %
   :alt: absolute value function

cube
^^^^

.. figure:: activation-cube.png
   :scale: 100 %
   :alt: cubic function

exp
^^^

.. figure:: activation-exp.png
   :scale: 100 %
   :alt: exponential function

expanded_log
^^^^^^^^^^^^^^

.. figure:: activation-expanded_log.png
    :scale: 100 %
    :alt: Expanded-range log function.

gauss
^^^^^

.. figure:: activation-gauss.png
   :scale: 100 %
   :alt: gaussian function

hat
^^^

.. figure:: activation-hat.png
   :scale: 100 %
   :alt: hat function

.. _identity-label:

identity
^^^^^^^^

.. figure:: activation-identity.png
   :scale: 100 %
   :alt: identity function

inv
^^^

.. figure:: activation-inv.png
   :scale: 100 %
   :alt: inverse (1/x) function

log
^^^

.. figure:: activation-log.png
   :scale: 100 %
   :alt: log function

sin
^^^

.. figure:: activation-sin.png
   :scale: 100 %
   :alt: sine function

skewed_log1p
^^^^^^^^^^^^

.. figure:: activation-skewed_log1p.png
    :scale: 100 %
    :alt: shifted log-plus function

square
^^^^^^

.. figure:: activation-square.png
   :scale: 100 %
   :alt: square function

CPPN-intended activation functions (multi-parameter)
---------------------------------------------------------------------------

.. _hat-gauss-label:

hat_gauss
^^^^^^^^^^^

.. figure:: activation-hat_gauss.png
    :scale: 100 %
    :alt: Weighted average of gauss and hat functions.

.. _scaled-expanded-log-label:

scaled_expanded_log
^^^^^^^^^^^^^^^^^^^^

.. figure:: activation-scaled_expanded_log.png
    :scale: 100 %
    :alt: A version of expanded_log with variable scaling (with partially-counterbalancing weights both inside and outside the expanded_log function).

.. _multiparam-log-inv-label:

multiparam_log_inv
^^^^^^^^^^^^^^^^^^^^

.. figure:: activation-multiparam_log_inv.png
    :scale: 100 %
    :alt: Above 0.0, equivalent to scaled_expanded_log with a+1.0; below, weighted mean with inv of -1*x.

.. _multiparam-pow-label:

multiparam_pow
^^^^^^^^^^^^^^^^^^^^

.. figure:: activation-multiparam_pow.png
    :scale: 100 %
    :alt: Above a=1, pow(z, a); below 1, pow(z, pow(2,(a-1.0))
