import finpy_tse as fpy

def import_data(symbols_input, symbols_start_date, symbols_end_date):
    results = {}

    for x in symbols_input:
        # Fetch USD/IRR exchange rate data from FinaPy
        if x == 'دلار':
            USD_RIAL = fpy.Get_USD_RIAL(
                start_date=symbols_start_date,
                end_date=symbols_end_date,
                ignore_date=False,
                show_weekday=False,
                double_date=False
            )
            results[x] = USD_RIAL
        # Fetch main index (CWI) data
        elif x == 'شاخص کل':
            CWI = fpy.Get_CWI_History(
                start_date=symbols_start_date,
                end_date=symbols_end_date,
                ignore_date=False,
                just_adj_close=False,
                show_weekday=False,
                double_date=False
            )
            results[x] = CWI
        # Fetch price history for a listed stock symbol on TSE
        else:
            STOCK = fpy.Get_Price_History(
                stock=x,
                start_date=symbols_start_date,
                end_date=symbols_end_date,
                ignore_date=False,
                adjust_price=False,
                show_weekday=False,
                double_date=False
            )
            results[x] = STOCK

    return results 
