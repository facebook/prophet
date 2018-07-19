# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from holidays import HolidayBase, MONDAY, WEEKEND, rd, easter
from lunardate import LunarDate
from calendar import Calendar
from datetime import date, timedelta
from convertdate.islamic import to_gregorian, from_gregorian
import warnings

# Official public holidays at a country level
# ------------ Holidays in Brazil---------------------
class Brazil(HolidayBase):
    """
    Implement public holidays in Brazil

    Reference:
    https://en.wikipedia.org/wiki/Public_holidays_in_Brazil
    """

    def __init__(self, **kwargs):
        self.country = "BR"
        HolidayBase.__init__(self, **kwargs)

    def _populate(self, year):
        # New Year's Day
        if not self.observed and date(year, 1, 1).weekday() in WEEKEND:
            pass
        else:
            self[date(year, 1, 1)] = "New Year's Day"

        # Tiradentes
        self[date(year, 4, 21)] = "Tiradentes"

        # Worker's Day
        self[date(year, 5, 1)] = "Worker's Day"

        # Independence Day
        self[date(year, 9, 7)] = "Independence Day"

        # Our Lady of the Apparition
        self[date(year, 10, 12)] = "Our Lady of the Apparition"

        # All Souls' Day
        self[date(year, 11, 2)] = "All Souls' Day"

        # Republic Proclamation Day
        self[date(year, 11, 5)] = "Republic Proclamation Day"

        # Christmas
        self[date(year, 12, 25)] = "Christmas"


class BR(Brazil):
    pass


# ------------ Holidays in Indonesia---------------------
class Indonesia(HolidayBase):
    """
    Implement public holidays in Indonesia

    Reference:
    https://en.wikipedia.org/wiki/Public_holidays_in_Indonesia

    Please note: Indonesia is a multi-cultural community and we only implement
    the national wide public holidays.
    """

    def __init__(self, **kwargs):
        self.country = "ID"
        HolidayBase.__init__(self, **kwargs)

    def _populate(self, year):
        # New Year's Day
        if not self.observed and date(year, 1, 1).weekday() in WEEKEND:
            pass
        else:
            self[date(year, 1, 1)] = "New Year's Day"

        # Chinese New Year/ Spring Festival
        name = "Chinese New Year"
        self[LunarDate(year, 1, 1).toSolarDate()] = name

        # Day of Silence / Nyepi
        # Note:
        # This holiday is determined by Balinese calendar, which is not currently
        # available. Only hard coded version of this holiday from 2009 to 2019
        # is available.
        warning_msg = "We only support Nyepi holiday from 2009 to 2019"
        warnings.warn(warning_msg, Warning)

        name = "Day of Silence/ Nyepi"
        if year == 2009:
            self[date(year, 3, 26)] = name
        elif year == 2010:
            self[date(year, 3, 16)] = name
        elif year == 2011:
            self[date(year, 3, 5)] = name
        elif year == 2012:
            self[date(year, 3, 23)] = name
        elif year == 2013:
            self[date(year, 3, 12)] = name
        elif year == 2014:
            self[date(year, 3, 31)] = name
        elif year == 2015:
            self[date(year, 3, 21)] = name
        elif year == 2016:
            self[date(year, 3, 9)] = name
        elif year == 2017:
            self[date(year, 3, 28)] = name
        elif year == 2018:
            self[date(year, 3, 17)] = name
        elif year == 2019:
            self[date(year, 3, 7)] = name
        else:
            pass

        # Ascension of the Prophet
        islam_year = from_gregorian(year, 3, 17)[0]
        y, m, d = to_gregorian(islam_year, 7, 27)
        name = "Ascension of the Prophet"
        self[date(year, m, d)] = name

        # Labor Day
        name = "Labor Day"
        self[date(year, 5, 1)] = name

        # Ascension of Jesus Christ
        name = "Ascension of Jesus"
        hdate = easter(year) + rd(days=+39)
        self[hdate] = name

        # Buddha's Birthday
        name = "Buddha's Birthday"
        self[LunarDate(year, 4, 15).toSolarDate()] = name

        # Pancasila Day, since 2017
        if year >= 2017:
            name = "Pancasila Day"
            self[date(year, 6, 1)] = name

        # Eid al-Fitr
        islam_year = from_gregorian(year, 6, 15)[0]
        y1, m1, d1 = to_gregorian(islam_year, 10, 1)
        y2, m2, d2 = to_gregorian(islam_year, 10, 2)
        name = "Eid al-Fitr"
        self[date(year, m1, d1)] = name
        self[date(year, m2, d2)] = name

        # Independence Day
        name = "Independence Day"
        self[date(year, 8, 17)] = name

        # Feast of the Sacrifice
        islam_year = from_gregorian(year, 8, 22)[0]
        y, m, d = to_gregorian(islam_year, 12, 10)
        name = "Feast of the Sacrifice"
        self[date(year, m, d)] = name

        # Islamic New Year
        islam_year = from_gregorian(year, 9, 11)[0]
        y, m, d = to_gregorian(islam_year + 1, 1, 1)
        name = "Islamic New Year"
        self[date(year, m, d)] = name

        # Birth of the Prophet
        islam_year = from_gregorian(year, 11, 20)[0]
        y, m, d = to_gregorian(islam_year + 1, 3, 12)
        name = "Birth of the Prophet"
        self[date(year, m, d)] = name

        # Christmas
        self[date(year, 12, 25)] = "Christmas"


class ID(Indonesia):
    pass


# ------------ Holidays in India---------------------
class India(HolidayBase):
    """
    Implement public holidays in India

    Reference:
    https://en.wikipedia.org/wiki/Public_holidays_in_India

    Please note:
    India is a culturally diverse and fervent society, celebrate various
    holidays and festivals. We only implement the holidays that **all states
    and territories** celebrate.
    """

    def __init__(self, **kwargs):
        self.country = "IN"
        HolidayBase.__init__(self, **kwargs)

    def _populate(self, year):
        # --------------------------------
        # Three national days
        #     Republic Day
        #     Independence Day
        #     Gandhi Jayanti
        # --------------------------------
        # Republic Day
        name = "Republic Day"
        self[date(year, 1, 26)] = name

        # Independence Day
        name = "Independence Day"
        self[date(year, 8, 15)] = name

        # Gandhi Jayanti
        name = "Gandhi Jayanti"
        self[date(year, 10, 2)] = name
        # --------------------------------
        # Hindo holidays
        #     Diwali
        #     Holi
        # --------------------------------

        # Diwali, Holi
        # http://www.theholidayspot.com/diwali/calendar.htm
        # https://www.timeanddate.com/holidays/india/diwali?starty=
        # https://www.infoplease.com/calendar-holidays/major-holidays/
        warning_msg = "We only support Diwali and Holi holidays from 2010 to 2025"
        warnings.warn(warning_msg, Warning)
        name1 = "Diwali"
        name2 = "Holi"
        if year == 2010:
            self[date(year, 12, 5)] = name1
            self[date(year, 2, 28)] = name2
        elif year == 2011:
            self[date(year, 10, 26)] = name1
            self[date(year, 3, 19)] = name2
        elif year == 2012:
            self[date(year, 11, 13)] = name1
            self[date(year, 3, 8)] = name2
        elif year == 2013:
            self[date(year, 11, 3)] = name1
            self[date(year, 3, 26)] = name2
        elif year == 2014:
            self[date(year, 10, 23)] = name1
            self[date(year, 3, 17)] = name2
        elif year == 2015:
            self[date(year, 11, 11)] = name1
            self[date(year, 3, 6)] = name2
        elif year == 2016:
            self[date(year, 10, 30)] = name1
            self[date(year, 3, 24)] = name2
        elif year == 2017:
            self[date(year, 10, 19)] = name1
            self[date(year, 3, 13)] = name2
        elif year == 2018:
            self[date(year, 11, 7)] = name1
            self[date(year, 3, 2)] = name2
        elif year == 2019:
            self[date(year, 10, 27)] = name1
            self[date(year, 3, 21)] = name2
        elif year == 2020:
            self[date(year, 11, 14)] = name1
            self[date(year, 3, 9)] = name2
        elif year == 2021:
            self[date(year, 11, 4)] = name1
            self[date(year, 3, 28)] = name2
        elif year == 2022:
            self[date(year, 10, 24)] = name1
            self[date(year, 3, 18)] = name2
        elif year == 2023:
            self[date(year, 10, 12)] = name1
            self[date(year, 3, 7)] = name2
        elif year == 2024:
            self[date(year, 11, 1)] = name1
            self[date(year, 3, 25)] = name2
        elif year == 2025:
            self[date(year, 10, 21)] = name1
            self[date(year, 3, 14)] = name2
        else:
            pass

        # --------------------------------
        # Islamic holidays
        #     Day of Ashura
        #     Mawlid
        #     Eid ul-Fitr
        #     Eid al-Adha
        # --------------------------------

        # Day of Ashura
        # 10th day of 1st Islamic month
        islam_year = from_gregorian(year, 10, 1)[0]
        y, m, d = to_gregorian(islam_year, 1, 10)
        name = "Day of Ashura"
        self[date(year, m, d)] = name

        # Mawlid, Birth of the Prophet
        # 12th day of 3rd Islamic month
        islam_year = from_gregorian(year, 11, 20)[0]
        y, m, d = to_gregorian(islam_year, 3, 12)
        name = "Mawlid"
        self[date(y, m, d)] = name

        # Eid ul-Fitr
        # 1st and 2nd day of 10th Islamic month
        islam_year = from_gregorian(year, 6, 15)[0]
        y1, m1, d1 = to_gregorian(islam_year, 10, 1)
        y2, m2, d2 = to_gregorian(islam_year, 10, 2)
        name = "Eid al-Fitr"
        self[date(year, m1, d1)] = name
        self[date(year, m2, d2)] = name

        # Eid al-Adha, i.e., Feast of the Sacrifice
        islam_year = from_gregorian(year, 8, 22)[0]
        y, m, d = to_gregorian(islam_year, 12, 10)
        name = "Feast of the Sacrifice"
        self[date(year, m, d)] = name

        # --------------------------------
        # Christian holidays
        #    New Year, Palm Sunday,
        #    Maundy Thursday
        #    Good Friday
        #    Easter Sunday
        #    Feast of Pentecost
        #    Fest of St. Theresa of Calcutta
        #    Feast of the Blessed Virgin
        #    All Saints Day
        #    All Souls Day
        #    Christmas Day
        #    Boxing Day
        #    Feast of Holy Family
        # --------------------------------
        # New Year's Day
        self[date(year, 1, 1)] = "New Year's Day"

        # Palm Sunday
        name = "Palm Sunday"
        self[easter(year) - rd(days=7)] = name

        # Maundy Thursday
        name = "Maundy Thursday"
        self[easter(year) - rd(days=3)] = name

        # Good Friday
        name = "Good Friday"
        self[easter(year) - rd(days=2)] = name

        # Easter Sunday
        name = "Easter Sunday"
        self[easter(year)] = name

        # Feast of Pentecost
        name = "Feast of Pentecost"
        self[easter(year) + rd(days=49)] = name

        # Fest of St. Theresa of Calcutta
        name = "Fest of St. Theresa of Calcutta"
        self[date(year, 9, 5)] = name

        # Feast of the Blessed Virgin
        name = "Feast of the Blessed Virgin"
        self[date(year, 9, 8)] = name

        # All Saints Day
        name = "All Saints Day"
        self[date(year, 11, 1)] = name

        # All Souls Day
        name = "All Souls Day"
        self[date(year, 11, 2)] = name

        # Christmas
        name = "Christmas Day"
        self[date(year, 12, 25)] = name

        # Boxing Day
        name = "Boxing Day"
        self[date(year, 12, 26)] = name

        # Feast of Holy Family
        name = "Feast of Holy Family"
        self[date(year, 12, 30)] = name


class IN(India):
    pass


# ------------ Holidays in Malaysia---------------------
class Malaysia(HolidayBase):
    """
    Implement public holidays in Malaysia

    Reference:
    https://en.wikipedia.org/wiki/Public_holidays_in_Malaysia
    """

    def __init__(self, **kwargs):
        self.country = "MY"
        HolidayBase.__init__(self, **kwargs)

    def _populate(self, year):
        # New Year's Day
        name = "New Year's Day"
        self[date(year, 1, 1)] = name

        # Birthday of Prophet, Mawlid in India
        # 12th day of 3rd Islamic month
        islam_year = from_gregorian(year, 11, 20)[0]
        y, m, d = to_gregorian(islam_year, 3, 12)
        name = "Birth of Prophet"
        self[date(y, m, d)] = name

        # Chinese New Year
        name = "Chinese New Year"
        self[LunarDate(year, 1, 1).toSolarDate()] = name

        # Tamil New Year
        # Note: it's not necessarily 04/14
        # due to the local calendar
        # other possible dates are 04/13 and 04/15
        name = "Tamil New Year"
        self[date(year, 4, 14)] = name

        # Good Friday
        name = "Good Friday"
        self[easter(year) - rd(days=2)] = name

        # Labor Day
        name = "Labor Day"
        self[date(year, 5, 1)] = name

        # Buddha's Birthday
        name = "Wesak Day"
        self[LunarDate(year, 4, 15).toSolarDate()] = name

        # King's birthday
        # https://www.thestar.com.my/news/nation/2017/04/26/
        # Before 2017: first Saturday of June
        # 2017-2021: last Saturday of July
        name = "King's birthday"
        if year < 2017:
            c = Calendar(firstweekday=MONDAY)
            monthcal = c.monthdatescalendar(year, 6)

            l1 = len(monthcal)
            saturdays = []
            for i in range(l1):
                if monthcal[i][5].month == 6:
                    saturdays.append(monthcal[i][5])
            self[saturdays[0]] = name
        elif (year >= 2017) and (year <= 2021):
            c = Calendar(firstweekday=MONDAY)
            monthcal = c.monthdatescalendar(year, 7)

            l1 = len(monthcal)
            saturdays = []
            for i in range(l1):
                if monthcal[i][5].month == 7:
                    saturdays.append(monthcal[i][5])
            self[saturdays[-1]] = name

        # Eid al-Fitr
        islam_year = from_gregorian(year, 6, 15)[0]
        y1, m1, d1 = to_gregorian(islam_year, 10, 1)
        y2, m2, d2 = to_gregorian(islam_year, 10, 2)
        name = "Eid al-Fitr"
        self[date(year, m1, d1)] = name
        self[date(year, m2, d2)] = name

        # Malaysia Day
        name = "Malaysia Day"
        self[date(year, 9, 16)] = name

        # Feast of the Sacrifice
        islam_year = from_gregorian(year, 8, 22)[0]
        y, m, d = to_gregorian(islam_year, 12, 10)
        name = "Feast of the Sacrifice"
        self[date(year, m, d)] = name

        # First Day of Muharram
        islam_year = from_gregorian(year, 9, 11)[0]
        y, m, d = to_gregorian(islam_year + 1, 1, 1)
        name = "First Day of Muharram"
        self[date(y, m, d)] = name

        # Christmas
        name = "Christmas Day"
        self[date(year, 12, 25)] = name


class MY(Malaysia):
    pass


# ------------ Holidays in Vietnam---------------------
class Vietnam(HolidayBase):
    """
    Implement public holidays in Vietnam

    Reference:
    https://en.wikipedia.org/wiki/Public_holidays_in_Vietnam
    """

    def __init__(self, **kwargs):
        self.country = "VN"
        HolidayBase.__init__(self, **kwargs)

    def _populate(self, year):
        # New Year's Day
        name = "New Year's Day"
        self[date(year, 1, 1)] = name

        # Vietnamese New Year
        name = "Vietnamese New Year"
        self[LunarDate(year - 1, 12, 30).toSolarDate()] = name
        self[LunarDate(year, 1, 1).toSolarDate()] = name
        self[LunarDate(year, 1, 2).toSolarDate()] = name
        self[LunarDate(year, 1, 3).toSolarDate()] = name
        self[LunarDate(year, 1, 4).toSolarDate()] = name
        self[LunarDate(year, 1, 5).toSolarDate()] = name

        # Hung Kings Commemorations
        name = "Hung Kings Commemorations"
        self[LunarDate(year, 3, 10).toSolarDate()] = name

        # Reunification Day
        name = "Reunification Day"
        self[date(year, 4, 30)] = name

        # Labor Day/International Workers' Day
        name = "Labor Day/International Workers' Day"
        self[date(year, 5, 1)] = name

        # National Day
        name = "National Day"
        self[date(year, 9, 2)] = name


class VN(Vietnam):
    pass


# ------------ Holidays in Thailand---------------------
class Thailand(HolidayBase):
    """
    Implement public holidays in Thailand

    Reference:
    https://en.wikipedia.org/wiki/Public_holidays_in_Thailand
    """

    def __init__(self, **kwargs):
        self.country = "TH"
        HolidayBase.__init__(self, **kwargs)

    def _populate(self, year):
        # New Year's Day
        name = "New Year's Day"
        self[date(year, 1, 1)] = name

        # Magha Pujab
        # Note:
        # This holiday is determined by Buddhist calendar, which is not currently
        # available. Only hard coded version of this holiday from 2016 to 2019
        # is available.

        name = "Magha Pujab/Makha Bucha"
        if year == 2016:
            self[date(year, 2, 22)] = name
        elif year == 2017:
            self[date(year, 2, 11)] = name
        elif year == 2018:
            self[date(year, 3, 1)] = name
        elif year == 2019:
            self[date(year, 2, 19)] = name
        else:
            pass

        # Chakri Memorial Day
        name = "Chakri Memorial Day"
        april_6 = date(year, 4, 6).weekday()
        if april_6 == 5:
            self[date(year, 4, 6 + 2)] = name
        elif april_6 == 6:
            self[date(year, 4, 6 + 1)] = name
        else:
            self[date(year, 4, 6)] = name

        # Songkran Festival
        name = "Songkran Festival"
        self[date(year, 4, 14)] = name

        # Royal Ploughing Ceremony
        # arbitrary day in May

        # Buddha's Birthday
        name = "Buddha's Birthday"
        self[LunarDate(year, 4, 15).toSolarDate()] = name

        # Coronation Day, removed in 2017
        name = "Coronation Day"
        if year < 2017:
            self[date(year, 5, 5)] = name

        # King Maha Vajiralongkorn's Birthday
        name = "King Maha Vajiralongkorn's Birthday"
        self[date(year, 7, 28)] = name

        # Asalha Puja
        # This is also a Buddha holiday, and we only implement
        # the hard coded version from 2006 to 2025
        # reference:
        # http://www.when-is.com/asalha_puja.asp
        warning_msg = "We only support Asalha Puja holiday from 2006 to 2025"
        warnings.warn(warning_msg, Warning)
        name = "Asalha Puja"
        if year == 2006:
            self[date(year, 7, 11)] = name
        elif year == 2007:
            self[date(year, 6, 30)] = name
        elif year == 2008:
            self[date(year, 7, 18)] = name
        elif year == 2009:
            self[date(year, 7, 7)] = name
        elif year == 2010:
            self[date(year, 7, 25)] = name
        elif year == 2011:
            self[date(year, 7, 15)] = name
        elif year == 2012:
            self[date(year, 8, 2)] = name
        elif year == 2013:
            self[date(year, 7, 30)] = name
        elif year == 2014:
            self[date(year, 7, 13)] = name
        elif year == 2015:
            self[date(year, 7, 30)] = name
        elif year == 2016:
            self[date(year, 7, 15)] = name
        elif year == 2017:
            self[date(year, 7, 9)] = name
        elif year == 2018:
            self[date(year, 7, 29)] = name
        elif year == 2019:
            self[date(year, 7, 16)] = name
        elif year == 2020:
            self[date(year, 7, 5)] = name
        elif year == 2021:
            self[date(year, 7, 24)] = name
        elif year == 2022:
            self[date(year, 7, 13)] = name
        elif year == 2023:
            self[date(year, 7, 3)] = name
        elif year == 2024:
            self[date(year, 7, 21)] = name
        elif year == 2025:
            self[date(year, 7, 10)] = name
        else:
            pass

        # Beginning of Vassa
        warning_msg = "We only support Vassa holiday from 2006 to 2020"
        warnings.warn(warning_msg, Warning)
        name = "Beginning of Vassa"
        if year == 2006:
            self[date(year, 7, 12)] = name
        elif year == 2007:
            self[date(year, 7, 31)] = name
        elif year == 2008:
            self[date(year, 7, 19)] = name
        elif year == 2009:
            self[date(year, 7, 8)] = name
        elif year == 2010:
            self[date(year, 7, 27)] = name
        elif year == 2011:
            self[date(year, 7, 16)] = name
        elif year == 2012:
            self[date(year, 8, 3)] = name
        elif year == 2013:
            self[date(year, 7, 23)] = name
        elif year == 2014:
            self[date(year, 7, 13)] = name
        elif year == 2015:
            self[date(year, 8, 1)] = name
        elif year == 2016:
            self[date(year, 7, 20)] = name
        elif year == 2017:
            self[date(year, 7, 9)] = name
        elif year == 2018:
            self[date(year, 7, 28)] = name
        elif year == 2019:
            self[date(year, 7, 17)] = name
        elif year == 2020:
            self[date(year, 7, 6)] = name
        else:
            pass

        # The Queen Sirikit's Birthday
        name = "The Queen Sirikit's Birthday"
        self[date(year, 8, 12)] = name

        # Anniversary for the Death of King Bhumibol Adulyadej
        name = "Anniversary for the Death of King Bhumibol Adulyadej"
        self[date(year, 10, 13)] = name

        # King Chulalongkorn Day
        name = "King Chulalongkorn Day"
        self[date(year, 10, 23)] = name

        # King Bhumibol Adulyadej's Birthday Anniversary
        name = "King Bhumibol Adulyadej's Birthday Anniversary"
        self[date(year, 12, 5)] = name

        # Constitution Day
        name = "Constitution Day"
        self[date(year, 12, 10)] = name

        # New Year's Eve
        name = "New Year's Eve"
        self[date(year, 12, 31)] = name


class TH(Thailand):
    pass


# ------------ Holidays in Philippines---------------------
class Philippines(HolidayBase):
    """
    Implement public holidays in Philippines

    Reference:
    https://en.wikipedia.org/wiki/Public_holidays_in_Thailand
    """

    def __init__(self, **kwargs):
        self.country = "PH"
        HolidayBase.__init__(self, **kwargs)

    def _populate(self, year):
        # New Year's Day
        name = "New Year's Day"
        self[date(year, 1, 1)] = name

        # Maundy Thursday
        name = "Maundy Thursday"
        self[easter(year) - rd(days=3)] = name

        # Good Friday
        name = "Good Friday"
        self[easter(year) - rd(days=2)] = name

        # Day of Valor
        name = "Day of Valor"
        self[date(year, 4, 9)] = name

        # Labor Day
        name = "Labor Day"
        self[date(year, 5, 1)] = name

        # Independence Day
        name = "Independence Day"
        self[date(year, 6, 12)] = name

        # Eid al-Fitr
        name = "Eid al-Fitr"
        islam_year = from_gregorian(year, 6, 15)[0]
        y1, m1, d1 = to_gregorian(islam_year, 10, 1)
        self[date(year, m1, d1) - timedelta(days=1)] = name

        # Eid al-Adha, i.e., Feast of the Sacrifice
        name = "Feast of the Sacrifice"
        islam_year = from_gregorian(year, 8, 22)[0]
        y, m, d = to_gregorian(islam_year, 12, 10)
        self[date(year, m, d)] = name

        # National Heroes' Day
        name = "National Heroes' Day"
        self[date(year, 8, 27)] = name

        # Bonifacio Day
        name = "Bonifacio Day"
        self[date(year, 11, 30)] = name

        # Christmas Day
        name = "Christmas Day"
        self[date(year, 12, 25)] = name

        # Rizal Day
        name = "Rizal Day"
        self[date(year, 12, 30)] = name


class PH(Philippines):
    pass


# ------------ Holidays in Turkey---------------------
class Turkey(HolidayBase):
    """
    Implement public holidays in Turkey

    Reference:
    https://en.wikipedia.org/wiki/Public_holidays_in_Thailand
    """

    def __init__(self, **kwargs):
        self.country = "TU"
        HolidayBase.__init__(self, **kwargs)

    def _populate(self, year):
        # New Year's Day
        name = "New Year's Day"
        self[date(year, 1, 1)] = name

        # National Sovereignty and Children's Day
        name = "National Sovereignty and Children's Day	"
        self[date(year, 4, 23)] = name

        # Labor Day
        name = "Labor Day"
        self[date(year, 5, 1)] = name

        # Commemoration of Atatürk, Youth and Sports Day
        name = "Commemoration of Atatürk, Youth and Sports Day"
        self[date(year, 5, 19)] = name

        # Democracy and National Unity Day
        name = "Democracy and National Unity Day"
        self[date(year, 7, 15)] = name

        # Victory Day
        name = "Victory Day"
        self[date(year, 8, 30)] = name

        # Republic Day
        name = "Republic Day"
        self[date(year, 10, 29)] = name

        # Eid al-Fitr
        name = "Eid al-Fitr"
        islam_year = from_gregorian(year, 6, 15)[0]
        y1, m1, d1 = to_gregorian(islam_year, 10, 1)
        self[date(year, m1, d1) - timedelta(days=1)] = name

        # Eid al-Adha, i.e., Feast of the Sacrifice
        name = "Feast of the Sacrifice"
        islam_year = from_gregorian(year, 8, 22)[0]
        y, m, d = to_gregorian(islam_year, 12, 10)
        self[date(year, m, d)] = name


class TU(Turkey):
    pass


# ------------ Holidays in Pakistan---------------------
class Pakistan(HolidayBase):
    """
    Implement public holidays in Pakistan

    Reference:
    https://en.wikipedia.org/wiki/Public_holidays_in_Pakistan
    """

    def __init__(self, **kwargs):
        self.country = "PK"
        HolidayBase.__init__(self, **kwargs)

    def _populate(self, year):

        # Kashmir Solidarity Day
        name = "Kashmir Solidarity Day"
        self[date(year, 2, 5)] = name

        # Pakistan Day
        name = "Pakistan Day"
        self[date(year, 3, 23)] = name

        # Labor Day
        name = "Labor Day"
        self[date(year, 5, 1)] = name

        # Independence Day
        name = "Independence Day"
        self[date(year, 8, 14)] = name

        # Iqbal Day
        name = "Iqbal Day"
        self[date(year, 11, 9)] = name

        # Christmas Day
        # Also birthday of PK founder
        name = "Christmas Day"
        self[date(year, 12, 25)] = name

        # Eid al-Adha, i.e., Feast of the Sacrifice
        name = "Feast of the Sacrifice"
        islam_year = from_gregorian(year, 8, 22)[0]
        y1, m1, d1 = to_gregorian(islam_year, 12, 10)
        y2, m2, d2 = to_gregorian(islam_year, 12, 11)
        y3, m3, d3 = to_gregorian(islam_year, 12, 12)
        self[date(year, m1, d1)] = name
        self[date(year, m2, d2)] = name
        self[date(year, m3, d3)] = name

        # Eid al-Fitr
        name = "Eid al-Fitr"
        islam_year = from_gregorian(year, 6, 15)[0]
        y1, m1, d1 = to_gregorian(islam_year, 10, 1)
        y2, m2, d2 = to_gregorian(islam_year, 10, 2)
        y3, m3, d3 = to_gregorian(islam_year, 10, 3)
        self[date(year, m1, d1)] = name
        self[date(year, m2, d2)] = name
        self[date(year, m3, d3)] = name

        # Mawlid, Birth of the Prophet
        # 12th day of 3rd Islamic month
        islam_year = from_gregorian(year, 11, 20)[0]
        y, m, d = to_gregorian(islam_year, 3, 12)
        name = "Mawlid"
        self[date(y, m, d)] = name

        # Day of Ashura
        # 10th and 11th days of 1st Islamic month
        islam_year = from_gregorian(year, 10, 1)[0]
        y1, m1, d1 = to_gregorian(islam_year, 1, 10)
        y2, m2, d2 = to_gregorian(islam_year, 1, 11)
        name = "Day of Ashura"
        self[date(year, m1, d1)] = name
        self[date(year, m2, d2)] = name

        # Shab e Mairaj
        islam_year = from_gregorian(year, 4, 13)[0]
        y1, m1, d1 = to_gregorian(islam_year, 7, 27)
        name = "Shab e Mairaj"
        self[date(year, m1, d1)] = name

        # Defence Day
        name = "Defence Day"
        self[date(year, 9, 6)] = name

        # Death Anniversary of Quaid-e-Azam
        name = "Death Anniversary of Quaid-e-Azam"
        self[date(year, 9, 11)] = name


class PK(Pakistan):
    pass


# ------------ Holidays in Bangladesh---------------------
class Bangladesh(HolidayBase):
    """
    Implement public holidays in Bangladesh

    Reference:
    https://en.wikipedia.org/wiki/Public_holidays_in_Bangladesh
    """

    def __init__(self, **kwargs):
        self.country = "BD"
        HolidayBase.__init__(self, **kwargs)

    def _populate(self, year):

        # Language Martyrs' Day
        name = "Language Martyrs' Day"
        self[date(year, 2, 21)] = name

        # Mujib's birthday
        name = "Mujib's birthday"
        self[date(year, 3, 17)] = name

        # Independence Day
        name = "Independence Day"
        self[date(year, 3, 26)] = name

        # Bengali New Year's Day
        name = "Bengali New Year's Day"
        self[date(year, 4, 14)] = name

        # Labor Day, May Day (local name)
        name = "Labor Day"
        self[date(year, 5, 1)] = name

        # National Mourning Day
        name = "National Mourning Day"
        self[date(year, 8, 15)] = name

        # Victory Day
        name = "Victory Day"
        self[date(year, 12, 16)] = name


class BD(Bangladesh):
    pass


# ------------ Holidays in Egypt---------------------
class Egypt(HolidayBase):
    """
    Implement public holidays in Egypt

    Reference:
    https://en.wikipedia.org/wiki/Public_holidays_in_Egypt
    """

    def __init__(self, **kwargs):
        self.country = "EG"
        HolidayBase.__init__(self, **kwargs)

    def _populate(self, year):

        # Fixed holidays
        # Christmas
        name = "Christmas"
        self[date(year, 1, 7)] = name

        # Revolution Day, after 2011
        name = "Revolution Day 2011"
        if year <= 2011:
            self[date(year, 1, 25)] = name

        # Sinai Liberation Day, after 1982
        name = "Sinai Liberation Day"
        if year <= 1982:
            self[date(year, 4, 25)] = name

        # Labor Day
        name = "Labor Day"
        self[date(year, 5, 1)] = name

        # Revolution Day
        name = "Sinai Liberation Day"
        self[date(year, 7, 23)] = name

        # Armed Forces Day
        name = "Armed Forces Day"
        self[date(year, 10, 6)] = name

        # Sham El Nessim
        # The Monday following Orthodox Easter
        name = "Sham El Nessim"
        orthodox_easter = easter(year, method=2)
        self[orthodox_easter + timedelta(days=1)] = name

        # Islamic New Year
        islam_year = from_gregorian(year, 9, 11)[0]
        y, m, d = to_gregorian(islam_year + 1, 1, 1)
        name = "Islamic New Year"
        self[date(year, m, d)] = name

        # Birthday of Prophet, Mawlid in India
        # 12th day of 3rd Islamic month
        islam_year = from_gregorian(year, 11, 20)[0]
        y, m, d = to_gregorian(islam_year, 3, 12)
        name = "Birth of Prophet"
        self[date(y, m, d)] = name

        # Eid ul-Fitr
        # 1st and 2nd day of 10th Islamic month
        islam_year = from_gregorian(year, 6, 15)[0]
        y1, m1, d1 = to_gregorian(islam_year, 10, 1)
        y2, m2, d2 = to_gregorian(islam_year, 10, 2)
        y3, m3, d3 = to_gregorian(islam_year, 10, 3)
        name = "Eid al-Fitr"
        self[date(year, m1, d1)] = name
        self[date(year, m2, d2)] = name
        self[date(year, m3, d3)] = name

        # Eid al-Adha, i.e., Feast of the Sacrifice
        islam_year = from_gregorian(year, 8, 22)[0]
        y1, m1, d1 = to_gregorian(islam_year, 12, 10)
        y2, m2, d2 = to_gregorian(islam_year, 12, 11)
        y3, m3, d3 = to_gregorian(islam_year, 12, 12)
        y4, m4, d4 = to_gregorian(islam_year, 12, 13)
        name = "Feast of the Sacrifice"
        self[date(year, m1, d1)] = name
        self[date(year, m2, d2)] = name
        self[date(year, m3, d3)] = name
        self[date(year, m4, d4)] = name


class EG(Egypt):
    pass


# ------------ Holidays in China---------------------
class China(HolidayBase):
    """
    Implement public holidays in China

    Reference:
    https://en.wikipedia.org/wiki/Public_holidays_in_China
    """

    def __init__(self, **kwargs):
        self.country = "CN"
        HolidayBase.__init__(self, **kwargs)

    def _populate(self, year):
        # New Year's Day
        name = "New Year's Day"
        self[date(year, 1, 1)] = name

        # Chinese New Year/ Spring Festival
        name = "Chinese New Year"
        self[LunarDate(year, 1, 1).toSolarDate()] = name

        # Tomb-Sweeping Day
        name = "Tomb-Sweeping Day"
        self[date(year, 4, 4)] = name
        self[date(year, 4, 5)] = name

        # Labor Day
        name = "Labor Day"
        self[date(year, 5, 1)] = name

        # Dragon Boat Festival
        name = "Dragon Boat Festival"
        self[LunarDate(year, 5, 5).toSolarDate()] = name

        # Mid-Autumn Festival
        name = "Mid-Autumn Festival"
        self[LunarDate(year, 8, 15).toSolarDate()] = name

        # National Day
        name = "National Day"
        self[date(year, 10, 1)] = name


class CN(China):
    pass

# ------------ Holidays in Russia---------------------
class Russia(HolidayBase):
    """
    Implement public holidays in Russia

    Reference:
    https://en.wikipedia.org/wiki/Public_holidays_in_Russia

    Please note:
    Orthodox Christmas Day is official day off at Russia
    But the Dec. 25 Christmas is also celebrated.
    """

    def __init__(self, **kwargs):
        self.country = "RU"
        HolidayBase.__init__(self, **kwargs)

    def _populate(self, year):
        # New Year's Day
        name = "New Year's Day"
        self[date(year, 1, 1)] = name

        # Orthodox Christmas day
        name = "Orthodox Christmas Day"
        self[date(year, 1, 7)] = name

        # Dec. 25 Christmas Day
        name = "Christmas Day"
        self[date(year, 12, 25)] = name

        # Defender of the Fatherland Day
        name = "Defender of the Fatherland Day"
        self[date(year, 2, 23)] = name

        # International Women's Day
        name = "International Women's Day"
        self[date(year, 3, 8)] = name

        # National Flag Day
        name = "National Flag Day"
        self[date(year, 8, 22)] = name

        # Spring and Labour Day
        name = "Spring and Labour Day"
        self[date(year, 5, 1)] = name

        # Victory Day
        name = "Victory Day"
        self[date(year, 5, 9)] = name

        # Russia Day
        name = "Russia Day"
        self[date(year, 6, 12)] = name

        # Unity Day
        name = "Unity Day"
        self[date(year, 11, 4)] = name


class RU(Russia):
    pass
