# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import warnings
from calendar import Calendar, MONDAY
from datetime import date, timedelta

from convertdate.islamic import from_gregorian, to_gregorian
from lunarcalendar import Lunar, Converter
from lunarcalendar.converter import DateNotExist

from holidays import WEEKEND, HolidayBase
from dateutil.easter import easter, EASTER_ORTHODOX
from dateutil.relativedelta import relativedelta as rd


class Brazil(HolidayBase):
    """Implements public holidays in Brazil."""

    def __init__(self, **kwargs):
        self.country = "BR"
        HolidayBase.__init__(self, **kwargs)

    def _populate(self, year):
        if not self.observed and date(year, 1, 1).weekday() in WEEKEND:
            pass
        else:
            self[date(year, 1, 1)] = "New Year's Day"

        self[date(year, 4, 21)] = "Tiradentes"

        self[date(year, 5, 1)] = "Labor Day"

        self[date(year, 9, 7)] = "Independence Day"

        self[date(year, 10, 12)] = "Our Lady of Apparecida's Day"

        self[date(year, 11, 2)] = "All Souls Day"

        self[date(year, 11, 15)] = "Proclamation of the Republic"

        self[date(year, 12, 25)] = "Christmas Day"

        self[easter(year) - rd(days=2)] = "Good Friday"

        self[easter(year)] = "Easter"

        # Corpus Christi is the first Thursday after Trinity Sunday.
        # Trinity Sunday is the first Sunday after Pentecost.
        # Pentecost is 50 days after Easter.
        # Easter is on Sunday.
        # So Pentecost is on Monday.
        # Trinity Sunday is easter + 56.
        # Corpus Christi is easter + 60.
        self[easter(year) + rd(days=50)] = "Pentecost"
        self[easter(year) + rd(days=56)] = "Trinity Sunday"
        self[easter(year) + rd(days=60)] = "Corpus Christi Holiday"

        self[easter(year) - rd(days=46)] = "Ash Wednesday"

        self[easter(year) - rd(days=46) - rd(weekday=TU(-1))] = "Rio Carnival"


class BR(Brazil):
    pass


class Indonesia(HolidayBase):
    """Implements public holidays in Indonesia."""

    def __init__(self, **kwargs):
        self.country = "ID"
        HolidayBase.__init__(self, **kwargs)

    def _populate(self, year):
        # New Year's Day, observed if falls on weekend.
        # If observed is False and it falls on weekend, it will not be observed.
        if not self.observed and date(year, 1, 1).weekday() in WEEKEND:
            pass
        else:
            self[date(year, 1, 1)] = "New Year's Day"

        for offset in range(-1, 2, 1):
            ds = Converter.Lunar2Solar(Lunar(year + offset, 1, 1)).to_date()
            if ds.year == year:
                self[ds] = "Chinese New Year"

        # Day of Silence / Nyepi / Hindu New Year is determined by Balinese calendar,
        # which becomes available every year.
        # Hard coded from 2009 to 2022.
        dates = {
            2009: date(2009, 3, 26),
            2010: date(2010, 3, 16),
            2011: date(2011, 3, 5),
            2012: date(2012, 3, 23),
            2013: date(2013, 3, 12),
            2014: date(2014, 3, 31),
            2015: date(2015, 3, 21),
            2016: date(2016, 3, 9),
            2017: date(2017, 3, 28),
            2018: date(2018, 3, 17),
            2019: date(2019, 3, 7),
            2020: date(2020, 3, 25),
            2021: date(2021, 3, 14),
            2022: date(2022, 3, 3)
        }
        if year in dates:
            self[dates[year]] = "Day of Silence/ Nyepi"
        else:
            warnings.warn(f"Day of Slience / Nyepi is not available for the selected year {year}.")

        # Falls on the 27th day of Rajab, the 7th month in the Islamic Calendar.
        for offset in range(-1, 2, 1):
            islam_year = from_gregorian(year + offset, 3, 17)[0]
            y, m, d = to_gregorian(islam_year, 7, 27)
            if y == year:
                self[date(y, m, d)] = "Ascension of the Prophet Muhammad"

        self[date(year, 5, 1)] = "Labor Day"

        # 39 days after Easter Sunday.
        self[easter(year) + rd(days=+39)] = "Ascension of Jesus"

        # Lunar calendar 4/15.
        for offset in range(-1, 2, 1):
            ds = Converter.Lunar2Solar(Lunar(year + offset, 4, 15)).to_date()
            if ds.year == year:
                self[ds] = "Buddha's Birthday"

        # Since 2017
        if year >= 2017:
            self[date(year, 6, 1)] = "Pancasila Day"

        for offset in range(-1, 2, 1):
            islam_year = from_gregorian(year + offset, 6, 15)[0]
            y1, m1, d1 = to_gregorian(islam_year, 10, 1)
            y2, m2, d2 = to_gregorian(islam_year, 10, 2)
            if y1 == year:
                self[date(y1, m1, d1)] = "Eid al-Fitr"
            if y2 == year:
                self[date(y2, m2, d2)] = "Eid al-Fitr"

        self[date(year, 8, 17)] = "Independence Day"

        for offset in range(-1, 2, 1):
            islam_year = from_gregorian(year + offset, 8, 22)[0]
            y, m, d = to_gregorian(islam_year, 12, 10)
            if y == year:
                self[date(y, m, d)] = "Feast of the Sacrifice"

        for offset in range(-1, 2, 1):
            islam_year = from_gregorian(year + offset, 9, 11)[0]
            y, m, d = to_gregorian(islam_year + 1, 1, 1)
            if y == year:
                self[date(y, m, d)] = "Islamic New Year"

        for offset in range(-1, 2, 1):
            islam_year = from_gregorian(year + offset, 11, 20)[0]
            y, m, d = to_gregorian(islam_year + 1, 3, 12)
            if y == year:
                self[date(y, m, d)] = "Birth of the Prophet"

        self[date(year, 12, 25)] = "Christmas"


class ID(Indonesia):
    pass


class India(HolidayBase):
    """Implements public holidays in India."""

    def __init__(self, **kwargs):
        self.country = "IN"
        HolidayBase.__init__(self, **kwargs)

    def _populate(self, year):
        self[date(year, 1, 26)] = "Republic Day"

        self[date(year, 8, 15)] = "Independence Day"

        self[date(year, 10, 2)] = "Gandhi Jayanti"

        self[date(year, 5, 1)] = "Labor Day"

        self[date(year, 1, 14)] = "Makar Sandranti"

        # Diwali and Holi are hard coded.
        # Available between 2010 and 2030.
        diwali_dates = {
            2010: date(2010, 12, 5),
            2011: date(2011, 10, 26),
            2012: date(2012, 11, 13),
            2013: date(2013, 11, 3),
            2014: date(2014, 10, 23),
            2015: date(2015, 11, 11),
            2016: date(2016, 10, 30),
            2017: date(2017, 10, 19),
            2018: date(2018, 11, 7),
            2019: date(2019, 10, 27),
            2020: date(2020, 11, 14),
            2021: date(2021, 11, 4),
            2022: date(2020, 10, 24),
            2023: date(2023, 10, 12),
            2024: date(2024, 11, 1),
            2025: date(2025, 10, 21),
            2026: date(2026, 11, 8),
            2027: date(2027, 10, 29),
            2028: date(2028, 10, 17),
            2029: date(2029, 11, 5),
            2030: date(2030, 10, 26)
        }
        holi_dates = {
            2010: date(2010, 2, 28),
            2011: date(2011, 3, 19),
            2012: date(2012, 3, 8),
            2013: date(2013, 3, 26),
            2014: date(2014, 3, 17),
            2015: date(2015, 3, 6),
            2016: date(2016, 3, 24),
            2017: date(2017, 3, 13),
            2018: date(2018, 3, 2),
            2019: date(2019, 3, 21),
            2020: date(2020, 3, 9),
            2021: date(2021, 3, 28),
            2022: date(2022, 3, 18),
            2023: date(2023, 3, 7),
            2024: date(2024, 3, 25),
            2025: date(2025, 3, 14),
            2026: date(2026, 3, 3),
            2027: date(2027, 3, 22),
            2028: date(2028, 3, 11),
            2029: date(2029, 2, 28),
            2030: date(2030, 3, 19)
        }
        if year in diwali_dates:
            self[diwali_dates[year]] = "Diwali"
        else:
            warnings.warn(f"Diwali is not available for year {year}.")
        if year in holi_dates:
            self[holi_dates[year]] = "Holi"
        else:
            warnings.warn(f"Holi is not available for year {year}.")

        # 10th day of 1st Islamic month
        for offset in range(-1, 2, 1):
            islam_year = from_gregorian(year + offset, 10, 1)[0]
            y, m, d = to_gregorian(islam_year, 1, 10)
            if y == year:
                self[date(y, m, d)] = "Day of Ashura"

        # 12th day of 3rd Islamic month
        for offset in range(-1, 2, 1):
            islam_year = from_gregorian(year + offset, 11, 20)[0]
            y, m, d = to_gregorian(islam_year, 3, 12)
            if y == year:
                self[date(y, m, d)] = "Mawlid"

        # 1st and 2nd day of 10th Islamic month
        for offset in range(-1, 2, 1):
            islam_year = from_gregorian(year + offset, 6, 15)[0]
            y1, m1, d1 = to_gregorian(islam_year, 10, 1)
            y2, m2, d2 = to_gregorian(islam_year, 10, 2)
            if y1 == year:
                self[date(y1, m1, d1)] = "Eid al-Fitr"
            if y2 == year:
                self[date(y2, m2, d2)] = "Eid al-Fitr"

        for offset in range(-1, 2, 1):
            islam_year = from_gregorian(year + offset, 8, 22)[0]
            y, m, d = to_gregorian(islam_year, 12, 10)
            if y == year:
                self[date(y, m, d)] = "Feast of the Sacrifice"

        self[date(year, 1, 1)] = "New Year's Day"

        for offset in range(-1, 2, 1):
            ds = easter(year + offset) - rd(days=7)
            if ds.year == year:
                self[ds] = "Palm Sunday"

        for offset in range(-1, 2, 1):
            ds = easter(year + offset) - rd(days=3)
            if ds.year == year:
                self[ds] = "Maundy Thursday"

        for offset in range(-1, 2, 1):
            ds = easter(year + offset) - rd(days=2)
            if ds.year == year:
                self[ds] = "Good Friday"

        for offset in range(-1, 2, 1):
            ds = easter(year + offset)
            if ds.year == year:
                self[ds] = "Easter Sunday"

        for offset in range(-1, 2, 1):
            ds = easter(year + offset) + rd(days=49)
            if ds.year == year:
                self[ds] = "Feast of Pentecost"

        self[date(year, 9, 5)] = "Fest of St. Theresa of Calcutta"

        self[date(year, 9, 8)] = "Feast of the Blessed Virgin"

        self[date(year, 11, 1)] = "All Saints Day"

        self[date(year, 11, 2)] = "All Souls Day"

        self[date(year, 12, 25)] = "Christmas Day"

        self[date(year, 12, 26)] = "Boxing Day"

        self[date(year, 12, 30)] = "Feast of Holy Family"


class IN(India):
    pass


class Malaysia(HolidayBase):
    """Implements public holidays in Malaysia."""

    def __init__(self, **kwargs):
        self.country = "MY"
        HolidayBase.__init__(self, **kwargs)

    def _populate(self, year):
        self[date(year, 1, 1)] = "New Year's Day"

        # 12th day of 3rd Islamic month
        for offset in range(-1, 2, 1):
            islam_year = from_gregorian(year + offset, 11, 20)[0]
            y, m, d = to_gregorian(islam_year, 3, 12)
            if y == year:
                self[date(y, m, d)] = "Birth of Prophet Muhammad"

        for offset in range(-1, 2, 1):
            ds = Converter.Lunar2Solar(Lunar(year + offset, 1, 1)).to_date()
            if ds.year == year:
                self[ds] = "Chinese New Year"

        # First day of Tamil month Chithirai.
        # Approximated.
        self[date(year, 4, 14)] = "Tamil New Year"

        for offset in range(-1, 2, 1):
            ds = easter(year + offset) - rd(days=2)
            if ds.year == year:
                self[ds] = "Good Friday"

        self[date(year, 5, 1)] = "Labor Day"

        for offset in range(-1, 2, 1):
            ds = Converter.Lunar2Solar(Lunar(year + offset, 4, 15)).to_date()
            if ds.year == year:
                self[ds] = "Wesak Day"

        # King's birthdays are different during different years.
        # < 2017: first Sat of Jun.
        # 2017: last Sat of July.
        # 2018 - 2019: 9/9.
        # > 2020: first Mon of Jun.
        if year < 2017:
            c = Calendar(firstweekday=MONDAY)
            monthcal = c.monthdatescalendar(year, 6)  # all dates in June in full weeks.

            for i in range(len(monthcal)):
                if monthcal[i][5].month == 6:  # checks if the Saturday is in June.
                    self[monthcal[i][5]] = "King's birthday"
                    break
        elif year == 2017:
            c = Calendar(firstweekday=MONDAY)
            monthcal = c.monthdatescalendar(year, 7)  # all dates in July in full weeks.

            for i in range(1, len(monthcal) + 1):
                if monthcal[-i][5].month == 7:  # checks if the Saturday is in July.
                    self[monthcal[i][5]] = "King's birthday"
                    break
        elif year in [2018, 2019]:
            self[date(year, 9, 9)] = "King's birthday"
        else:
            # The date may change in the future, but for now it is the first Monday of June.
            c = Calendar(firstweekday=MONDAY)
            monthcal = c.monthdatescalendar(year, 6)  # all dates in June in full weeks.

            for i in range(len(monthcal)):
                if monthcal[i][0].month == 6:  # checks if the Monday is in June.
                    self[monthcal[i][0]] = "King's birthday"
                    break

        for offset in range(-1, 2, 1):
            islam_year = from_gregorian(year + offset, 6, 15)[0]
            y1, m1, d1 = to_gregorian(islam_year, 10, 1)
            y2, m2, d2 = to_gregorian(islam_year, 10, 2)
            if y1 == year:
                self[date(y1, m1, d1)] = "Eid al-Fitr"
            if y2 == year:
                self[date(y2, m2, d2)] = "Eid al-Fitr"

        self[date(year, 9, 16)] = "Malaysia Day"

        for offset in range(-1, 2, 1):
            islam_year = from_gregorian(year + offset, 8, 22)[0]
            y, m, d = to_gregorian(islam_year, 12, 10)
            if y == year:
                self[date(y, m, d)] = "Feast of the Sacrifice"

        for offset in range(-1, 2, 1):
            islam_year = from_gregorian(year + offset, 9, 11)[0]
            y, m, d = to_gregorian(islam_year + 1, 1, 1)
            if y == year:
                self[date(y, m, d)] = "First Day of Muharram"

        self[date(year, 12, 25)] = "Christmas Day"


class MY(Malaysia):
    pass


class Vietnam(HolidayBase):
    """Implements public holidays in Vietnam."""

    def __init__(self, **kwargs):
        self.country = "VN"
        HolidayBase.__init__(self, **kwargs)

    def _populate(self, year):
        self[date(year, 1, 1)] = "New Year's Day"

        # Lasts for 6 days.
        for offset in range(-1, 2, 1):
            try:
                ds = Converter.Lunar2Solar(Lunar(year - 1 + offset, 12, 30)).to_date()
            except DateNotExist:
                ds = Converter.Lunar2Solar(Lunar(year - 1 + offset, 12, 29)).to_date()
            if ds.year == year:
                self[ds] = "Vietnamese New Year Eve"
            ds = Converter.Lunar2Solar(Lunar(year + offset, 1, 1)).to_date()
            if ds.year == year:
                self[ds] = "Vietnamese New Year"
            ds = Converter.Lunar2Solar(Lunar(year + offset, 1, 2)).to_date()
            if ds.year == year:
                self[ds] = "Vietnamese New Year"
            ds = Converter.Lunar2Solar(Lunar(year + offset, 1, 3)).to_date()
            if ds.year == year:
                self[ds] = "Vietnamese New Year"
            ds = Converter.Lunar2Solar(Lunar(year + offset, 1, 4)).to_date()
            if ds.year == year:
                self[ds] = "Vietnamese New Year"
            ds = Converter.Lunar2Solar(Lunar(year + offset, 1, 5)).to_date()
            if ds.year == year:
                self[ds] = "Vietnamese New Year"

        # Becomes a national holiday since 2007.
        if year >= 2007:
            for offset in range(-1, 2, 1):
                ds = Converter.Lunar2Solar(Lunar(year + offset, 3, 10)).to_date()
                if ds.year == year:
                    self[ds] = "Hung Kings Commemorations Day"

        self[date(year, 4, 30)] = "Reunification Day"

        self[date(year, 5, 1)] = "Labor Day"

        self[date(year, 9, 2)] = "Independence Day"


class VN(Vietnam):
    pass


class Thailand(HolidayBase):
    """Implements public holidays in Thailand."""

    def __init__(self, **kwargs):
        self.country = "TH"
        HolidayBase.__init__(self, **kwargs)

    def _populate(self, year):
        self[date(year, 1, 1)] = "New Year's Day"

        # Magha Pujab is determined by Buddhist calendar.
        # Hard coded from 2016 to 2022.
        dates = {
            2016: date(2016, 2, 22),
            2017: date(2017, 2, 11),
            2018: date(2018, 3, 1),
            2019: date(2019, 2, 19),
            2020: date(2020, 2, 10),
            2021: date(2021, 2, 26),
            2022: date(2022, 2, 16)
        }
        if year in dates:
            self[dates[year]] = "Magha Puja"
        else:
            warnings.warn(f"Magha Puja is not available for year {year}.")

        # If April 6th falls on a weekend, the following Monday will be taken.
        april_6 = date(year, 4, 6).weekday()
        if april_6 == 5:
            self[date(year, 4, 6 + 2)] = "Chakri Memorial Day"
        elif april_6 == 6:
            self[date(year, 4, 6 + 1)] = "Chakri Memorial Day"
        else:
            self[date(year, 4, 6)] = "Chakri Memorial Day"

        self[date(year, 4, 13)] = "Songkran Festival"
        self[date(year, 4, 14)] = "Songkran Festival"
        self[date(year, 4, 15)] = "Songkran Festival"

        for offset in range(-1, 2, 1):
            ds = Converter.Lunar2Solar(Lunar(year + offset, 4, 15)).to_date()
            if ds.year == year:
                self[ds] = "Buddha's Birthday"

        # Coronation Day is removed in 2017
        if year < 2017:
            self[date(year, 5, 5)] = "Coronation Day"

        self[date(year, 7, 28)] = "King Maha Vajiralongkorn's Birthday"

        # Asalha Puja is a Buddha holiday.
        # Hard coded from 2006 to 2025
        dates = {
            2006: date(2006, 7, 11),
            2007: date(2007, 6, 30),
            2008: date(2008, 7, 18),
            2009: date(2009, 7, 7),
            2010: date(2010, 7, 25),
            2011: date(2011, 7, 15),
            2012: date(2012, 8, 2),
            2013: date(2013, 7, 30),
            2014: date(2014, 7, 13),
            2015: date(2015, 7, 30),
            2016: date(2016, 7, 15),
            2017: date(2017, 7, 9),
            2018: date(2018, 7, 29),
            2019: date(2019, 7, 16),
            2020: date(2020, 7, 5),
            2021: date(2021, 7, 24),
            2022: date(2022, 7, 13),
            2023: date(2023, 7, 3),
            2024: date(2024, 7, 21),
            2025: date(2025, 7, 10)
        }
        if year in dates:
            self[dates[year]] = "Asalha Puja"
        else:
            warnings.warn(f"Asalha Puja is not available for year {year}.")

        # Beginning of Vassa is harded coded between 2006 and 2021.
        dates = {
            2006: date(2006, 7, 12),
            2007: date(2007, 7, 31),
            2008: date(2008, 7, 19),
            2009: date(2009, 7, 8),
            2010: date(2010, 7, 27),
            2011: date(2011, 7, 16),
            2012: date(2012, 8, 3),
            2013: date(2013, 7, 23),
            2014: date(2014, 7, 13),
            2015: date(2015, 8, 1),
            2016: date(2016, 7, 20),
            2017: date(2017, 7, 9),
            2018: date(2018, 7, 28),
            2019: date(2019, 7, 17),
            2020: date(2020, 7, 6),
            2021: date(2021, 7, 25)
        }
        if year in dates:
            self[dates[year]] = "Beginning of Vassa"
        else:
            warnings.warn(f"Beginning of Vassa is not available for year {year}.")

        self[date(year, 8, 12)] = "The Queen Sirikit's Birthday"

        self[date(year, 10, 13)] = "Anniversary for the Death of King Bhumibol Adulyadej"

        self[date(year, 10, 23)] = "King Chulalongkorn Day"

        self[date(year, 12, 5)] = "King Bhumibol Adulyadej's Birthday Anniversary"

        self[date(year, 12, 10)] = "Constitution Day"

        self[date(year, 12, 31)] = "New Year's Eve"


class TH(Thailand):
    pass


class Philippines(HolidayBase):
    """Implements public holidays in Philippines."""

    def __init__(self, **kwargs):
        self.country = "PH"
        HolidayBase.__init__(self, **kwargs)

    def _populate(self, year):
        self[date(year, 1, 1)] = "New Year's Day"

        for offset in range(-1, 2, 1):
            ds = easter(year + offset) - rd(days=3)
            if ds.year == year:
                self[ds] = "Maundy Thursday"

        for offset in range(-1, 2, 1):
            ds = easter(year + offset) - rd(days=2)
            if ds.year == year:
                self[ds] = "Good Friday"

        self[date(year, 4, 9)] = "Day of Valor"

        self[date(year, 5, 1)] = "Labor Day"

        self[date(year, 6, 12)] = "Independence Day"

        # Philippines only observes one day on Eid al-Fitr.
        for offset in range(-1, 2, 1):
            islam_year = from_gregorian(year + offset, 6, 15)[0]
            y, m, d = to_gregorian(islam_year, 10, 1)
            ds = date(y, m, d) - timedelta(days=1)
            if ds.year == year:
                self[ds] = "Eid al-Fitr"

        for offset in range(-1, 2, 1):
            islam_year = from_gregorian(year + offset, 8, 22)[0]
            y, m, d = to_gregorian(islam_year, 12, 10)
            if y == year:
                self[date(y, m, d)] = "Feast of the Sacrifice"

        # Last Monday of August.
        c = Calendar(firstweekday=MONDAY)
        monthcal = c.monthdatescalendar(year, 8)  # all dates in August in full weeks.

        for i in range(1, len(monthcal) + 1):
            if monthcal[-i][0].month == 8:  # checks if the Monday is in August.
                self[monthcal[i][0]] = "National Heroes' Day"
                break

        self[date(year, 11, 30)] = "Bonifacio Day"

        self[date(year, 12, 25)] = "Christmas Day"

        self[date(year, 12, 30)] = "Rizal Day"


class PH(Philippines):
    pass


class Pakistan(HolidayBase):
    """Implements public holidays in Pakistan."""

    def __init__(self, **kwargs):
        self.country = "PK"
        HolidayBase.__init__(self, **kwargs)

    def _populate(self, year):
        self[date(year, 2, 5)] = "Kashmir Solidarity Day"

        self[date(year, 3, 23)] = "Pakistan Day"

        self[date(year, 5, 1)] = "Labor Day"

        self[date(year, 8, 14)] = "Independence Day"

        self[date(year, 11, 9)] = "Iqbal Day"

        # Also birthday of PK founder
        self[date(year, 12, 25)] = "Christmas Day"

        for offset in range(-1, 2, 1):
            islam_year = from_gregorian(year + offset, 8, 22)[0]
            y1, m1, d1 = to_gregorian(islam_year, 12, 10)
            y2, m2, d2 = to_gregorian(islam_year, 12, 11)
            y3, m3, d3 = to_gregorian(islam_year, 12, 12)
            if y1 == year:
                self[date(y1, m1, d1)] = "Feast of the Sacrifice"
            if y2 == year:
                self[date(y2, m2, d2)] = "Feast of the Sacrifice"
            if y3 == year:
                self[date(y3, m3, d3)] = "Feast of the Sacrifice"

        for offset in range(-1, 2, 1):
            islam_year = from_gregorian(year + offset, 6, 15)[0]
            y1, m1, d1 = to_gregorian(islam_year, 10, 1)
            y2, m2, d2 = to_gregorian(islam_year, 10, 2)
            y3, m3, d3 = to_gregorian(islam_year, 10, 3)
            if y1 == year:
                self[date(y1, m1, d1)] = "Eid al-Fitr"
            if y2 == year:
                self[date(y2, m2, d2)] = "Eid al-Fitr"
            if y3 == year:
                self[date(y3, m3, d3)] = "Eid al-Fitr"

        # 12th day of 3rd Islamic month
        for offset in range(-1, 2, 1):
            islam_year = from_gregorian(year + offset, 11, 20)[0]
            y, m, d = to_gregorian(islam_year, 3, 12)
            if y == year:
                self[date(y, m, d)] = "Mawlid"

        # 10th and 11th days of 1st Islamic month
        for offset in range(-1, 2, 1):
            islam_year = from_gregorian(year + offset, 10, 1)[0]
            y1, m1, d1 = to_gregorian(islam_year, 1, 10)
            y2, m2, d2 = to_gregorian(islam_year, 1, 11)
            if y1 == year:
                self[date(y1, m1, d1)] = "Day of Ashura"
            if y2 == year:
                self[date(y2, m2, d2)] = "Day of Ashura"

        # 27th day of the month of Rajab, the 7th month in the Islamic calendar.
        for offset in range(-1, 2, 1):
            islam_year = from_gregorian(year + offset, 4, 13)[0]
            y, m, d = to_gregorian(islam_year, 7, 27)
            if y == year:
                self[date(y, m, d)] = "Shab e Mairaj"

        self[date(year, 9, 6)] = "Defence Day"

        self[date(year, 9, 11)] = "Death Anniversary of Quaid-e-Azam"


class PK(Pakistan):
    pass


class Bangladesh(HolidayBase):
    """Implements public holidays in Bangladesh."""

    def __init__(self, **kwargs):
        self.country = "BD"
        HolidayBase.__init__(self, **kwargs)

    def _populate(self, year):
        self[date(year, 2, 21)] = "Language Martyrs' Day"

        self[date(year, 3, 17)] = "Mujib's birthday"

        self[date(year, 3, 26)] = "Independence Day"

        self[date(year, 4, 14)] = "Bengali New Year's Day"
        self[date(year, 4, 15)] = "Bengali New Year's Day"

        self[date(year, 5, 1)] = "Labor Day"

        self[date(year, 8, 15)] = "National Mourning Day"

        self[date(year, 12, 16)] = "Victory Day"


class BD(Bangladesh):
    pass


class Egypt(HolidayBase):
    """Implements public holidays in Egypt."""

    def __init__(self, **kwargs):
        self.country = "EG"
        HolidayBase.__init__(self, **kwargs)

    def _populate(self, year):
        self[date(year, 1, 1)] = "New Year's Day"

        self[date(year, 1, 7)] = "Coptic Christmas"

        # This revolution Day is only after 2011.
        if year >= 2011:
            self[date(year, 1, 25)] = "Revolution Day 2011"

        # Sinai Liberation Day is only after 1982.
        if year >= 1982:
            self[date(year, 4, 25)] = "Sinai Liberation Day"

        self[date(year, 5, 1)] = "Labor Day"

        self[date(year, 7, 23)] = "Revolution Day"

        self[date(year, 10, 6)] = "Armed Forces Day"

        # The Monday following Orthodox Easter
        for offset in range(-1, 2, 1):
            orthodox_easter = easter(year + offset, method=EASTER_ORTHODOX)
            ds = orthodox_easter + timedelta(days=1)
            if ds.year == year:
                self[ds] = "Sham El Nessim"

        for offset in range(-1, 2, 1):
            islam_year = from_gregorian(year + offset, 9, 11)[0]
            y, m, d = to_gregorian(islam_year + 1, 1, 1)
            if y == year:
                self[date(y, m, d)] = "Islamic New Year"

        # 12th day of 3rd Islamic month
        for offset in range(-1, 2, 1):
            islam_year = from_gregorian(year + offset, 11, 20)[0]
            y, m, d = to_gregorian(islam_year, 3, 12)
            if y == year:
                self[date(y, m, d)] = "Birth of Prophet"

        # 1st and 2nd day of 10th Islamic month
        for offset in range(-1, 2, 1):
            islam_year = from_gregorian(year + offset, 6, 15)[0]
            y1, m1, d1 = to_gregorian(islam_year, 10, 1)
            y2, m2, d2 = to_gregorian(islam_year, 10, 2)
            y3, m3, d3 = to_gregorian(islam_year, 10, 3)
            if y1 == year:
                self[date(y1, m1, d1)] = "Eid al-Fitr"
            if y2 == year:
                self[date(y2, m2, d2)] = "Eid al-Fitr"
            if y3 == year:
                self[date(y3, m3, d3)] = "Eid al-Fitr"

        for offset in range(-1, 2, 1):
            islam_year = from_gregorian(year + offset, 8, 22)[0]
            y1, m1, d1 = to_gregorian(islam_year, 12, 10)
            y2, m2, d2 = to_gregorian(islam_year, 12, 11)
            y3, m3, d3 = to_gregorian(islam_year, 12, 12)
            y4, m4, d4 = to_gregorian(islam_year, 12, 13)
            if y1 == year:
                self[date(y1, m1, d1)] = "Feast of the Sacrifice"
            if y2 == year:
                self[date(y2, m2, d2)] = "Feast of the Sacrifice"
            if y3 == year:
                self[date(y3, m3, d3)] = "Feast of the Sacrifice"
            if y4 == year:
                self[date(y4, m4, d4)] = "Feast of the Sacrifice"


class EG(Egypt):
    pass


class China(HolidayBase):
    """Implements public holidays in China."""

    def __init__(self, **kwargs):
        self.country = "CN"
        HolidayBase.__init__(self, **kwargs)

    def _populate(self, year):
        self[date(year, 1, 1)] = "New Year's Day"

        # Observes 3 days holidays.
        for offset in range(-1, 2, 1):
            ds = Converter.Lunar2Solar(Lunar(year + offset, 1, 1)).to_date()
            if ds.year == year:
                self[ds] = "Chinese New Year"
            ds = Converter.Lunar2Solar(Lunar(year + offset, 1, 2)).to_date()
            if ds.year == year:
                self[ds] = "Chinese New Year"
            ds = Converter.Lunar2Solar(Lunar(year + offset, 1, 3)).to_date()
            if ds.year == year:
                self[ds] = "Chinese New Year"

        self[date(year, 4, 4)] = "Tomb-Sweeping Day"
        self[date(year, 4, 5)] = "Tomb-Sweeping Day"

        self[date(year, 5, 1)] = "Labor Day"

        for offset in range(-1, 2, 1):
            ds = Converter.Lunar2Solar(Lunar(year + offset, 5, 5)).to_date()
            if ds.year == year:
                self[ds] = "Dragon Boat Festival"

        for offset in range(-1, 2, 1):
            ds = Converter.Lunar2Solar(Lunar(year + offset, 8, 15)).to_date()
            if ds.year == year:
                self[ds] = "Mid-Autumn Festival"

        # Observes 3 days holidays.
        self[date(year, 10, 1)] = "National Day"
        self[date(year, 10, 2)] = "National Day"
        self[date(year, 10, 3)] = "National Day"


class CN(China):
    pass


class Russia(HolidayBase):
    """Implements public holidays in Russia."""

    def __init__(self, **kwargs):
        self.country = "RU"
        HolidayBase.__init__(self, **kwargs)

    def _populate(self, year):
        self[date(year, 1, 1)] = "New Year's Day"
        self[date(year, 1, 2)] = "New Year's Day"
        self[date(year, 1, 3)] = "New Year's Day"
        self[date(year, 1, 4)] = "New Year's Day"
        self[date(year, 1, 5)] = "New Year's Day"
        self[date(year, 1, 6)] = "New Year's Day"

        self[date(year, 1, 7)] = "Orthodox Christmas Day"

        self[date(year, 12, 25)] = "Christmas Day"

        self[date(year, 2, 23)] = "Defender of the Fatherland Day"

        self[date(year, 3, 8)] = "International Women's Day"

        self[date(year, 8, 22)] = "National Flag Day"

        self[date(year, 5, 1)] = "Spring and Labour Day"

        self[date(year, 5, 9)] = "Victory Day"

        self[date(year, 6, 12)] = "Russia Day"

        self[date(year, 11, 4)] = "Unity Day"


class RU(Russia):
    pass


class Belarus(HolidayBase):
    """Implements public holidays in Belarus."""

    def __init__(self, **kwargs):
        self.country = "BY"
        HolidayBase.__init__(self, **kwargs)

    def _populate(self, year):
        self[date(year, 1, 1)] = "New Year's Day"

        self[date(year, 1, 7)] = "Orthodox Christmas Day"

        self[date(year, 3, 8)] = "International Women's Day"

        self[easter(year, EASTER_ORTHODOX) + timedelta(days=9)] = "Commemoration Day"

        self[date(year, 5, 1)] = "Spring and Labour Day"

        self[date(year, 5, 9)] = "Victory Day"

        self[date(year, 7, 3)] = "Independence Day"

        self[date(year, 11, 7)] = "October Revolution Day"

        self[date(year, 12, 25)] = "Christmas Day"


class BY(Belarus):
    pass


class UnitedArabEmirates(HolidayBase):
    """Implements public holidays in United Arab Emirates."""

    def __init__(self, **kwargs):
        self.country = "AE"
        HolidayBase.__init__(self, **kwargs)

    def _populate(self, year):
        self[date(year, 1, 1)] = "New Year's Day"

        for offset in range(-1, 2, 1):
            islam_year = from_gregorian(year + offset, 6, 15)[0]
            y1, m1, d1 = to_gregorian(islam_year, 9, 29)
            y2, m2, d2 = to_gregorian(
                islam_year, 9, 30
            )
            y3, m3, d3 = to_gregorian(islam_year, 10, 1)
            y4, m4, d4 = to_gregorian(islam_year, 10, 2)
            y5, m5, d5 = to_gregorian(islam_year, 10, 3)
            if y1 == year:
                self[date(y1, m1, d1)] = "Eid al-Fitr"
            if y2 == year:
                self[date(y2, m2, d2)] = "Eid al-Fitr"
            if y3 == year:
                self[date(y3, m3, d3)] = "Eid al-Fitr"
            if y4 == year:
                self[date(y4, m4, d4)] = "Eid al-Fitr"
            if y5 == year:
                self[date(y5, m5, d5)] = "Eid al-Fitr"

        for offset in range(-1, 2, 1):
            islam_year = from_gregorian(year + offset, 8, 22)[0]
            y, m, d = to_gregorian(islam_year, 12, 9)
            if y == year:
                self[date(y, m, d)] = "Day of Arafah"

        for offset in range(-1, 2, 1):
            islam_year = from_gregorian(year + offset, 8, 22)[0]
            y1, m1, d1 = to_gregorian(islam_year, 12, 10)
            y2, m2, d2 = to_gregorian(islam_year, 12, 11)
            y3, m3, d3 = to_gregorian(islam_year, 12, 12)
            if y1 == year:
                self[date(y1, m1, d1)] = "Feast of the Sacrifice"
            if y2 == year:
                self[date(y2, m2, d2)] = "Feast of the Sacrifice"
            if y3 == year:
                self[date(y3, m3, d3)] = "Feast of the Sacrifice"

        for offset in range(-1, 2, 1):
            islam_year = from_gregorian(year + offset, 9, 11)[0]
            y, m, d = to_gregorian(islam_year + 1, 1, 1)
            if y == year:
                self[date(y, m, d)] = "Islamic New Year"

        self[date(year, 11, 30)] = "Commemoration Day"

        self[date(year, 12, 2)] = "National Day"
        self[date(year, 12, 3)] = "National Day"


class AE(UnitedArabEmirates):
    pass


class Georgia(HolidayBase):
    """Implements public holidays in Georgia."""

    def __init__(self, **kwargs):
        self.country = "GE"
        HolidayBase.__init__(self, **kwargs)

    def _populate(self, year):
        self[date(year, 1, 1)] = "New Year's Day"

        self[date(year, 1, 2)] = "Second day of the New Year"

        self[date(year, 1, 7)] = "Orthodox Christmas"

        self[date(year, 1, 19)] = "Baptism Day of our Lord Jesus Christ"

        self[date(year, 3, 3)] = "Mother's Day"

        self[date(year, 3, 8)] = "International Women's Day"

        self[easter(year, EASTER_ORTHODOX) - timedelta(days=2)] = "Good Friday"

        self[easter(year, EASTER_ORTHODOX) - timedelta(days=1)] = "Great Saturday"

        self[easter(year, EASTER_ORTHODOX)] = "Easter Sunday"

        self[easter(year, EASTER_ORTHODOX) + timedelta(days=1)] = "Easter Monday"

        self[date(year, 4, 9)] = "National Unity Day"

        self[date(year, 5, 9)] = "Victory Day"

        self[date(year, 5, 12)] = "Saint Andrew the First-Called Day"

        self[date(year, 5, 26)] = "Independence Day"

        self[date(year, 8, 28)] = "Saint Mary's Day"

        self[date(year, 10, 14)] = "Day of Svetitskhoveli Cathedral"

        self[date(year, 12, 23)] = "Saint George's Day"


class GE(Georgia):
    pass
