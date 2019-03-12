"""
This module provides a collection of computation functions for the analysis of a "loan tape".

The computation functions all expect as input a Pandas dataframe, with columns:
   loan_id (str)
   original_principal_balance (float)
   loan_origination_date (date)
   loan_term (int)
   loan_apr (float)
   payment_amount (float)
   payment_date (date)
If your dataframe contains these columns, but with types that differ from above, the 'enforceDataConstraints(...)'
function can be called on your dataframe, to return a dataframe with the above types enforced.

The computation functions all return a Pandas dataframe, containing the applicable computation results.
"""

import pandas as pd
import numpy as np
import datetime
from dateutil import relativedelta
from scipy.optimize import fsolve


class _g:
    """
    This class provides a set of globals to the module.
    These globals are private to the module.  They should not be referenced outside it.
    """

    # Expected data column names.  Subsequent references in code should be to the global name.
    ID_COL                 = "loan_id"
    ORIGINAL_PRINCIPAL_COL = "original_principal_balance"
    ORIGINATION_DATE_COL   = "loan_origination_date"
    TERM_COL               = "loan_term"
    APR_COL                = "loan_apr"
    PAYMENT_AMOUNT_COL     = "payment_amount"
    PAYMENT_DATE_COL       = "payment_date"

    CUM_PRINCIPAL_COL = "cum_payment_towards_principal"
    CUM_INTEREST_COL  = "cum_payment_towards_interest"
    CUM_FEES_COL      = "cum_payment_towards_fees"

    VINTAGE_DATE               = "vintage_date"
    VINTAGE_PRINCIPAL          = "vintage_cumulative_principal"
    PAYMENT_TO_CREDIT_FACILITY = "monthly_payment_to_credit_facility"
    VINTAGE_CASHFLOW_FOR_IRR   = "vintage_cashflow_for_irr"

    # Expected ordering of the data columns.
    EXPECTED_COLS = [
        ID_COL,
        ORIGINAL_PRINCIPAL_COL,
        ORIGINATION_DATE_COL,
        TERM_COL,
        APR_COL,
        PAYMENT_AMOUNT_COL,
        PAYMENT_DATE_COL
    ]


def enforceDataConstraints(inputDataframe):
    """
    This function checks whether an input dataframe meets the expected columns and datatypes.  If expected datatypes
    are not met, this function enforces them.

    If data constraints cannot be enforced, this function will raise an exception, to notify the caller.
    This function **is not exhaustive**.  It provides "reasonable effort" data validation, on behalf of the user.

    INPUT   --> A dataframe with columns as specified in the module description.
    RETURNS --> A dataframe with columns as specified in the module description, with all datatypes enforced.
    
    **NOTE** This function is called by all module computation functions, prior to computation.  If the user wishes, they
    can run this function on their dataframes themselves, *before* inputting them to the computation functions.  This will
    speed up the computation functions, as the call they make to enforceDataConstraints(...) would then be a no-op.
    """

    # Deep-copy the input dataframe.  We will perform data constraints on this copy, transforming and returning it.
    df = inputDataframe.copy()

    # Check whether the input dataframe has the expected columns, in the expected order.
    # TODO: The constraint on expected order can possibly be eliminated, going forward.
    for dfColumn, expectedColumn in zip(df.columns, _g.EXPECTED_COLS):
        if dfColumn != expectedColumn:
            raise Exception(f"Input dataframe columns are {df.columns.values}; does not meet expected {_g.EXPECTED_COLS}")

    try:
        if df[_g.ORIGINAL_PRINCIPAL_COL].dtype != 'float64':
            parsedCol = df[_g.ORIGINAL_PRINCIPAL_COL].str.replace('$','').str.replace(',','')
            df[_g.ORIGINAL_PRINCIPAL_COL] = pd.to_numeric(parsedCol)
    except Exception as e:
        raise Exception(f"Error attempting to enforce float64 data constraint on column {_g.ORIGINAL_PRINCIPAL_COL}.")
    try:
        # This is NOOP if data is already datetime type.
        df[_g.ORIGINATION_DATE_COL] = pd.to_datetime(df[_g.ORIGINATION_DATE_COL], errors='coerce')
    except Exception as e:
        raise Exception(f"Error attempting to enforce datetime data constraint on column {_g.ORIGINATION_DATE_COL}.")
    try:
        if df[_g.TERM_COL].dtype != 'int64':
            df[_g.TERM_COL] = df[_g.TERM_COL].astype('int64')
    except Exception as e:
        raise(f"Error attempting to enforce int64 data constraint on column {_g.TERM_COL}.")
    try:
        if df[_g.APR_COL].dtype != 'float64':
            df[_g.APR_COL] = df[_g.APR_COL].astype('float64')
    except Exception as e:
        raise(f"Error attempting to enforce float64 data constraint on column {_g.APR_COL}.")
    try:
        if df[_g.PAYMENT_AMOUNT_COL].dtype != 'float64':
            parsedCol = df[_g.PAYMENT_AMOUNT_COL].str.replace('$','').str.replace(',','')
            df[_g.PAYMENT_AMOUNT_COL] = pd.to_numeric(parsedCol)
    except Exception as e:
        raise(f"Error attempting to enforce float64 data constraint on column {_g.PAYMENT_AMOUNT_COL}.")
    try:
        # This is NOOP if data is already datetime type.
        df[_g.PAYMENT_DATE_COL] = pd.to_datetime(df[_g.PAYMENT_DATE_COL], errors='coerce')
    except Exception as e:
        raise(f"Error attempting to enforce datetime data constraint on column {_g.PAYMENT_DATE_COL}.")

    # Drop any rows with NAN values in any of their columns.  This will drop data that comes in in unexpected form
    # (for example, a non-date value in a date column).  If that data is important, then it needs to be marshalled
    # into expected form upstream of this module.
    df.dropna(inplace=True)
    return df


def computeBreakdownOf_FeesInterestPrincipal(inputDataframe, aggregateOver=None):
    """
    This function provides the ability to compute the amount of payment per loan that went to each of
    (1) principal, (2) interest, (3) fees.

    The function will raise an exception in the event of data issue or error.

    INPUT   --> A dataframe with columns as specified in the module description.
    RETURNS --> A dataframe with columns as follows:
       loan_id (str)
       original_principal_balance (currency)  --> Dropped if 'aggregateOver' specified.
       loan_origination_date (date)           --> Dropped if 'aggregateOver' specified.
       loan_term (int)                        --> Dropped if 'aggregateOver' specified.
       loan_apr (float)                       --> Dropped if 'aggregateOver' specified.
       cum_payment_towards_principal (float)
       cum_payment_towards_interest (float)
       cum_payment_towards_fees (float)

    Options for aggregating the results are provided per the parameter 'aggregateOver'.  A handful of often-used
    aggregations are supported:
       'monthlyVintage' --> aggregates results for all loans originated in the same month

    If the user requires an aggregation of results that is not supported, 'aggregateOver' should be left empty, the
    resulting dataframe will contain results for all loans, and ad-hoc aggregation can be conducted on it by the user.
    """

    def computeFeesInterestPrincipal_ForLoan(dfForLoan):
        # Over the life of the loan, these are the values to be computed and returned by this function.
        cumPrincipal, cumInterest, cumFees = 0, 0, 0

        # Do some up-front validation that the data takes the form expected.
        loanID = dfForLoan[_g.ID_COL].iloc[0]
        if dfForLoan[_g.ORIGINAL_PRINCIPAL_COL].unique().size != 1:
            raise Exception(f"Column={_g.ORIGINAL_PRINCIPAL_COL} expected to have unique value for loanID={loanID}.")
        if dfForLoan[_g.ORIGINATION_DATE_COL].unique().size != 1:
            raise Exception(f"Column={_g.ORIGINATION_DATE_COL} expected to have unique value for loanID={loanID}.")
        if dfForLoan[_g.TERM_COL].unique().size != 1:
            raise Exception(f"Column={_g.TERM_COL} expected to have unique value for loanID={loanID}.")
        if dfForLoan[_g.APR_COL].unique().size != 1:
            raise Exception(f"Column={_g.APR_COL} expected to have unique value for loanID={loanID}.")

        # Get necessary data from dataframe columns.  Accessing the data thusly preserves the dtype.
        originationDate   = dfForLoan[_g.ORIGINATION_DATE_COL].iloc[0]
        termNumMonths     = dfForLoan[_g.TERM_COL].iloc[0]
        apr               = dfForLoan[_g.APR_COL].iloc[0]
        originalPrincipal = dfForLoan[_g.ORIGINAL_PRINCIPAL_COL].iloc[0]
        numMonthsRemaining = termNumMonths
        principalLedger    = originalPrincipal
        interestLedger     = 0
        feesLedger         = 0

        # Aggregate any rows with the same payment date.
        paymentTS = dfForLoan.groupby(_g.PAYMENT_DATE_COL).agg({_g.PAYMENT_AMOUNT_COL : 'sum'})
        paymentTS.sort_values(_g.PAYMENT_DATE_COL, inplace=True)  # Chronological order, past --> present

        priorPaymentDate     = originationDate
        paymentDeadlineDate  = originationDate + pd.DateOffset(months=1) + pd.DateOffset(days=13)
        targetMonthlyPayment = principalLedger * (apr/12 / (1 - (1 + apr/12)**(-numMonthsRemaining)))
        paymentLedger   = 0
        for index, row in paymentTS.iterrows():
            # From each row, get the date the payment was made, and the cumulative amount of the payment.
            paymentDate   = row.name
            paymentAmount = row[_g.PAYMENT_AMOUNT_COL]
            
            # Compute the accrued simple interest.  This is done on a daily basis, non-compounding. 
            daysSinceLastPayment = (paymentDate - priorPaymentDate).days
            interestAmountPerDay = apr/360.0 * principalLedger
            interestLedger      += interestAmountPerDay * daysSinceLastPayment

            # Need to determine whether to assess fees, and how much to assess.
            if paymentDate > paymentDeadlineDate:
                # If *multiple* pay periods have been missed since lass payment, assess a fee for each missed.
                # Again, EXACT methodology here would need to be determined from reading loan-provider's methodology.
                monthsPastDeadline = relativedelta.relativedelta(paymentDate, paymentDeadlineDate).months + 1
                for missedDeadline in range(monthsPastDeadline):
                    outstandingBalance = principalLedger + interestLedger + feesLedger
                    feesLedger += 0.05*outstandingBalance if 0.05*outstandingBalance > 20 else 20

            paymentAmountDist = paymentAmount
            if paymentAmountDist > feesLedger:
                paymentAmountDist -= feesLedger
                cumFees       += feesLedger
                feesLedger     = 0
                if paymentAmountDist > interestLedger:
                    paymentAmountDist -= interestLedger
                    cumInterest   += interestLedger
                    interestLedger = 0
                    if paymentAmountDist > principalLedger:
                        # Register the overpayment as a data anomaly?  For now, print to stdout, if overpayment by $5+
                        if paymentAmountDist - principalLedger >= 5:
                            print(f"!!! Payments for loanID={loanID} went over principal by ${paymentAmountDist-principalLedger:.2f}!!!")
                        paymentAmountDist  -= principalLedger
                        cumPrincipal   += principalLedger
                        principalLedger = 0
                        break
                    else:
                        principalLedger -= paymentAmountDist
                        cumPrincipal    += paymentAmountDist
                else:
                    interestLedger -= paymentAmountDist
                    cumInterest    += paymentAmountDist
            else:
                feesLedger -= paymentAmountDist
                cumFees    += paymentAmountDist

            # At this point in the code, we've registered the fees and interest that may have accrued since our last
            # payment, we've registered the payment amount, and we've counted the payment amount against any outstanding
            # (1) fees, (2) interest, (3) principal, in that order.
            #
            # Now, we need to address the payment ledger, and the targeted monthly payment.  If this payment exceeds
            # the last recorded payment deadline date, then this payment now counts against a new deadline, with a
            # new targeted monthly payment.
            #
            if paymentDate > paymentDeadlineDate:
                # Any existing paymentLedger prior to this tardy payment should be wiped clean.
                paymentLedger = 0
                # Need to determine whether *multiple* pay periods have been missed since last payment.
                monthsPastDeadline   = relativedelta.relativedelta(paymentDate, paymentDeadlineDate).months
                paymentDeadlineDate += pd.DateOffset(months=monthsPastDeadline+1)
                numMonthsRemaining  -= monthsPastDeadline+1
                if numMonthsRemaining <= 0: break  # If the loan has reached full term, break.
                targetMonthlyPayment = principalLedger * (apr/12 / (1 - (1 + apr/12)**(-numMonthsRemaining))) \
                                       + (interestLedger + feesLedger) / numMonthsRemaining
            paymentLedger += paymentAmount
            if paymentLedger >= targetMonthlyPayment:
                # Our paymentLedger has grown sufficiently large to meet our target monthly payment.  Thus, we
                # zero out the paymentLedger.
                paymentLedger = 0
                paymentDeadlineDate += pd.DateOffset(months=1)
                numMonthsRemaining  -= 1
                if numMonthsRemaining <= 0: break  # If the loan has reached full term, break.
                targetMonthlyPayment = principalLedger * (apr/12 / (1 - (1 + apr/12)**(-numMonthsRemaining))) \
                                       + (interestLedger + feesLedger) / numMonthsRemaining

            # Wind forward the prior payment date to this current payment.
            priorPaymentDate = paymentDate

        #--------End of for-loop-------------------------------------------------------------
        
        return pd.DataFrame({
            _g.ID_COL                 : [loanID],
            _g.ORIGINAL_PRINCIPAL_COL : [originalPrincipal],
            _g.ORIGINATION_DATE_COL   : [originationDate],
            _g.TERM_COL               : [termNumMonths],
            _g.APR_COL                : [apr],
            _g.CUM_PRINCIPAL_COL      : [cumPrincipal],
            _g.CUM_INTEREST_COL       : [cumInterest],
            _g.CUM_FEES_COL           : [cumFees]
        })
    #----------End of computeFeesInterestPrincipal_ForLoan(...) function----------------------

    df = enforceDataConstraints(inputDataframe)
    # TODO: Would find loans that are incomplete here, and either screen them out, or apply projection model.
    results = df.groupby(_g.ID_COL).apply(lambda dfForLoan: computeFeesInterestPrincipal_ForLoan(dfForLoan))
    results.reset_index(drop=True, inplace=True)  # Drop the top-level index created by groupby operation, and re-index.
    if aggregateOver == 'monthlyVintage':
        # Aggregates all loans that originated in the same year & month, summing over the columns.
        results.index = results[_g.ORIGINATION_DATE_COL]
        aggregated    = results.resample('M').sum()
        # An artifact of .resample('M') is an insertion of missing months.  Remove these.
        maskOutZeros = (aggregated[[_g.CUM_PRINCIPAL_COL, _g.CUM_INTEREST_COL, _g.CUM_FEES_COL]] == 0).all(axis=1)
        results      = aggregated[~maskOutZeros].reset_index()
        # We summed over all columns.  Drop those for which the sum does not make sense.
        results.drop([_g.ORIGINAL_PRINCIPAL_COL, _g.TERM_COL, _g.APR_COL], axis=1, inplace=True)
    return results


def computeIRR_ForMonthlyVintage(inputDataframe):
    """
    This function provides the ability to compute, for each monthly vintage (i.e. the collection of loans originated
    in the same month), the internal rate of return (IRR).

    The function will raise an exception in the event of data issue or error.

    INPUT   --> A dataframe with columns as specified in the module description.
    RETURNS --> A dataframe with columns as follows:
       vintage_date (date)
       vintage_irr (float)

    Using the formula for net present value,
      NPV = sum( Ct / (1+r)^t ), over t=0 to t=T
    where Ct is the cash inflow during period t (C0 is our initial outlay of cash, and thus is negative), t is
    the number of time periods (our time periods are monthly) out from the monthly vintage date, and r is the
    discount rate.  The IRR is the discount rate r at which NPV becomes 0.
    Each Ct is determined as the lesser of (A) the cumulative payments made during that period to the lender,
    or (B) the lender's payment to us during that period.
    """

    df = enforceDataConstraints(inputDataframe)
    # TODO: Would find loans that are incomplete here, and either screen them out, or apply projection model.

    # Here, we keep a single row for each loanID.  Then, we group on month of origination date (the monthly vintage),
    # and sum the total principal paid out for that month.
    # The grouping on month is done by indexing on the date we group over, using .resample('M'), then resetting the
    # index.  An artifact of .resample('M') is an insertion of missing months, with zeros.  We remove these.
    cashPaidOut       = df.drop_duplicates(_g.ID_COL)[[_g.ORIGINATION_DATE_COL, _g.ORIGINAL_PRINCIPAL_COL]]
    cashPaidOut.index = cashPaidOut[_g.ORIGINATION_DATE_COL]
    cashPaidOut       = cashPaidOut.resample('M').sum().reset_index()
    cashPaidOut       = cashPaidOut[cashPaidOut[_g.ORIGINAL_PRINCIPAL_COL] != 0]
    cashPaidOut       = cashPaidOut.rename(
                            {_g.ORIGINATION_DATE_COL   : _g.VINTAGE_DATE,
                             _g.ORIGINAL_PRINCIPAL_COL : _g.VINTAGE_PRINCIPAL},
                             axis='columns'
                        )

    # Here, we group on month of origination date (the monthly loan vintage), then compute the cashflow timeseries
    # over all loans in that group.
    # The grouping on month is done by indexing on the date we group over, using .resample('M'), then resetting the
    # index.  An artifact of .resample('M') is an insertion of missing months, with zeros.  We remove these.
    def computeCashflows_ForLoanGroup(dfForMonthlyVintage):
        # The input to this function is a dataframe containing the payment timeseries for a monthly vintage.
        # Here, we group on month of payment, then sum the total payments made during that month; this yields,
        # for the specified monthly vintage, a timeseries of cumulative monthly payments.
        # The grouping on month is done by indexing on the date we group over, using .resample('M'), then resetting
        # the index.  An artifact of .resample('M') is an insertion of missing months, with zeros.  This is desired
        # behavior: for any month in which no payments were made, we want the payment amount column to register as $0.
        cashPaidIn_OverTime       = dfForMonthlyVintage[[_g.PAYMENT_DATE_COL, _g.PAYMENT_AMOUNT_COL]]
        cashPaidIn_OverTime.index = cashPaidIn_OverTime[_g.PAYMENT_DATE_COL]
        cashPaidIn_OverTime       = cashPaidIn_OverTime.resample('M').sum()
        return cashPaidIn_OverTime

    cashPaidIn       = df
    cashPaidIn.index = cashPaidIn[_g.ORIGINATION_DATE_COL]
    cashPaidIn       = cashPaidIn.resample('M').apply(lambda dfForMonthlyVintage: computeCashflows_ForLoanGroup(dfForMonthlyVintage)).reset_index()
    cashPaidIn       = cashPaidIn[cashPaidIn[_g.PAYMENT_DATE_COL] != 0]
    cashPaidIn       = cashPaidIn.rename(
                           {_g.ORIGINATION_DATE_COL : _g.VINTAGE_DATE},
                           axis='columns'
                       )

    # Now, we need to take the cashPaidOut, and for each monthly vintage, determine the monthly payment made to
    # our credit facility by the loan provider, on the loan they took to make the monthly vintage.
    def computeMonthlyPayment(vintagePrincipal):
        aprOfLoan  = 0.14
        termOfLoan = 12
        return vintagePrincipal * (aprOfLoan/12 / (1 - (1 + aprOfLoan/12)**(-termOfLoan)))
    cashPaidOut[_g.PAYMENT_TO_CREDIT_FACILITY] = cashPaidOut[_g.VINTAGE_PRINCIPAL].apply(computeMonthlyPayment)

    # Our views on per-vintage out-cashflows vs. in-cashflows are now complete.  We are now prepared to use them
    # to compute IRR, for each vintage.  We do so below, and return the result.
    def computeIRR(dfForMonthlyVintage):
        initialCashOutlay = dfForMonthlyVintage[_g.VINTAGE_PRINCIPAL].unique()[0]
        cashFlows         = dfForMonthlyVintage[_g.VINTAGE_CASHFLOW_FOR_IRR].values
        cashFlows         = np.insert(cashFlows, 0, [-1*initialCashOutlay])
        monthIndex        = [i for i in range(0, np.size(cashFlows))]
        # NOTE that in 'computeCashflows_ForLoanGroup(...)', we used $0 placeholders for any month without payments.
        # This ensures that the monthIndex will not have gaps.
        def irrDecimal(cashFlows, monthIndex):
            def npv(irr, cashFlows, periodIndex):  
                return np.sum(cashFlows / (1. + irr) ** periodIndex)
            irr = fsolve(npv, x0=0.1, args=(cashFlows, monthIndex))
            return np.asscalar(irr)
        return irrDecimal(cashFlows, monthIndex) * 100  # Map from decimal to %
    # For a given vintage, for each timeslice, we compare the cashflow paid to the lender by loan recipients and the
    # cashflow paid to us by the lender.  At each timeslice, the lesser of the two cashflows is used in our IRR computation.
    composite = pd.merge(cashPaidOut, cashPaidIn, on=[_g.VINTAGE_DATE])
    composite[_g.VINTAGE_CASHFLOW_FOR_IRR] = composite.apply(
        lambda row: row[_g.PAYMENT_AMOUNT_COL] if row[_g.PAYMENT_AMOUNT_COL] < row[_g.PAYMENT_TO_CREDIT_FACILITY] else row[_g.PAYMENT_TO_CREDIT_FACILITY],
        axis='columns')
    vintageIRR = composite.groupby(_g.VINTAGE_DATE).apply(lambda dfForMonthlyVintage: computeIRR(dfForMonthlyVintage))

    return vintageIRR
