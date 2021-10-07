# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import warnings
from datetime import date, timedelta

from convertdate.islamic import from_gregorian, to_gregorian
from lunarcalendar import Lunar, Converter

from holidays import WEEKEND, HolidayBase, Turkey
from dateutil.easter import easter, EASTER_ORTHODOX
from dateutil.relativedelta import relativedelta as rd


# Official public holidays at a country level
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
        for offset in range(-1, 2, 1):
            ds = Converter.Lunar2Solar(Lunar(year + offset, 1, 1)).to_date()
            if ds.year == year:
                self[ds] = name

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
        name = "Ascension of the Prophet"
        for offset in range(-1, 2, 1):
            islam_year = from_gregorian(year + offset, 3, 17)[0]
            y, m, d = to_gregorian(islam_year, 7, 27)
            if y == year:
                self[date(y, m, d)] = name

        # Labor Day
        name = "Labor Day"
        self[date(year, 5, 1)] = name

        # Ascension of Jesus Christ
        name = "Ascension of Jesus"
        for offset in range(-1, 2, 1):
            ds = easter(year + offset) + rd(days=+39)
            if ds.year == year:
                self[ds] = name

        # Buddha's Birthday
        name = "Buddha's Birthday"
        for offset in range(-1, 2, 1):
            ds = Converter.Lunar2Solar(Lunar(year + offset, 4, 15)).to_date()
            if ds.year == year:
                self[ds] = name

        # Pancasila Day, since 2017
        if year >= 2017:
            name = "Pancasila Day"
            self[date(year, 6, 1)] = name

        # Eid al-Fitr
        name = "Eid al-Fitr"
        for offset in range(-1, 2, 1):
            islam_year = from_gregorian(year + offset, 6, 15)[0]
            y1, m1, d1 = to_gregorian(islam_year, 10, 1)
            y2, m2, d2 = to_gregorian(islam_year, 10, 2)
            if y1 == year:
                self[date(y1, m2, d2)] = name
            if y2 == year:
                self[date(y2, m2, d2)] = name

        # Independence Day
        name = "Independence Day"
        self[date(year, 8, 17)] = name

        # Feast of the Sacrifice
        name = "Feast of the Sacrifice"
        for offset in range(-1, 2, 1):
            islam_year = from_gregorian(year + offset, 8, 22)[0]
            y, m, d = to_gregorian(islam_year, 12, 10)
            if y == year:
                self[date(y, m, d)] = name

        # Islamic New Year
        name = "Islamic New Year"
        for offset in range(-1, 2, 1):
            islam_year = from_gregorian(year + offset, 9, 11)[0]
            y, m, d = to_gregorian(islam_year + 1, 1, 1)
            if y == year:
                self[date(y, m, d)] = name

        # Birth of the Prophet
        name = "Birth of the Prophet"
        for offset in range(-1, 2, 1):
            islam_year = from_gregorian(year + offset, 11, 20)[0]
            y, m, d = to_gregorian(islam_year + 1, 3, 12)
            if y == year:
                self[date(y, m, d)] = name

        # Christmas
        self[date(year, 12, 25)] = "Christmas"


class ID(Indonesia):
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
        for offset in range(-1, 2, 1):
            ds = Converter.Lunar2Solar(Lunar(year + offset, 4, 15)).to_date()
            if ds.year == year:
                self[ds] = name

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
        for offset in range(-1, 2, 1):
            ds = easter(year + offset) - rd(days=3)
            if ds.year == year:
                self[ds] = name

        # Good Friday
        name = "Good Friday"
        for offset in range(-1, 2, 1):
            ds = easter(year + offset) - rd(days=2)
            if ds.year == year:
                self[ds] = name

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
        for offset in range(-1, 2, 1):
            islam_year = from_gregorian(year + offset, 6, 15)[0]
            y, m, d = to_gregorian(islam_year, 10, 1)
            ds = date(y, m, d) - timedelta(days=1)
            if ds.year == year:
                self[ds] = name

        # Eid al-Adha, i.e., Feast of the Sacrifice
        name = "Feast of the Sacrifice"
        for offset in range(-1, 2, 1):
            islam_year = from_gregorian(year + offset, 8, 22)[0]
            y, m, d = to_gregorian(islam_year, 12, 10)
            if y == year:
                self[date(y, m, d)] = name

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
# This is now in Holidays, but with alias TR instead of the TU that we used.
# Include TU as an alias for backwards compatibility.


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
        for offset in range(-1, 2, 1):
            islam_year = from_gregorian(year + offset, 8, 22)[0]
            y1, m1, d1 = to_gregorian(islam_year, 12, 10)
            y2, m2, d2 = to_gregorian(islam_year, 12, 11)
            y3, m3, d3 = to_gregorian(islam_year, 12, 12)
            if y1 == year:
                self[date(y1, m1, d1)] = name
            if y2 == year:
                self[date(y2, m2, d2)] = name
            if y3 == year:
                self[date(y3, m3, d3)] = name

        # Eid al-Fitr
        name = "Eid al-Fitr"
        for offset in range(-1, 2, 1):
            islam_year = from_gregorian(year + offset, 6, 15)[0]
            y1, m1, d1 = to_gregorian(islam_year, 10, 1)
            y2, m2, d2 = to_gregorian(islam_year, 10, 2)
            y3, m3, d3 = to_gregorian(islam_year, 10, 3)
            if y1 == year:
                self[date(y1, m1, d1)] = name
            if y2 == year:
                self[date(y2, m2, d2)] = name
            if y3 == year:
                self[date(y3, m3, d3)] = name

        # Mawlid, Birth of the Prophet
        # 12th day of 3rd Islamic month
        name = "Mawlid"
        for offset in range(-1, 2, 1):
            islam_year = from_gregorian(year + offset, 11, 20)[0]
            y, m, d = to_gregorian(islam_year, 3, 12)
            if y == year:
                self[date(y, m, d)] = name

        # Day of Ashura
        # 10th and 11th days of 1st Islamic month
        name = "Day of Ashura"
        for offset in range(-1, 2, 1):
            islam_year = from_gregorian(year + offset, 10, 1)[0]
            y1, m1, d1 = to_gregorian(islam_year, 1, 10)
            y2, m2, d2 = to_gregorian(islam_year, 1, 11)
            if y1 == year:
                self[date(y1, m1, d1)] = name
            if y2 == year:
                self[date(y2, m2, d2)] = name

        # Shab e Mairaj
        name = "Shab e Mairaj"
        for offset in range(-1, 2, 1):
            islam_year = from_gregorian(year + offset, 4, 13)[0]
            y, m, d = to_gregorian(islam_year, 7, 27)
            if y == year:
                self[date(y, m, d)] = name

        # Defence Day
        name = "Defence Day"
        self[date(year, 9, 6)] = name

        # Death Anniversary of Quaid-e-Azam
        name = "Death Anniversary of Quaid-e-Azam"
        self[date(year, 9, 11)] = name


class PK(Pakistan):
    pass


# ------------ Holidays in Belarus---------------------
class Belarus(HolidayBase):
    """
    Implement public holidays in Belarus

    Reference:
    https://en.wikipedia.org/wiki/Public_holidays_in_Belarus

    Please note:
    Some holidays might collide with weekends and therefore not compensated with next business day
    as International Women's Day
    """

    def __init__(self, **kwargs):
        self.country = "BY"
        HolidayBase.__init__(self, **kwargs)

    def _populate(self, year):
        # New Year's Day
        name = "New Year's Day"
        self[date(year, 1, 1)] = name

        # Orthodox Christmas day
        name = "Orthodox Christmas Day"
        self[date(year, 1, 7)] = name

        # International Women's Day
        name = "International Women's Day"
        self[date(year, 3, 8)] = name

        # Commemoration Day
        name = "Commemoration Day"
        self[easter(year, EASTER_ORTHODOX) + timedelta(days=9)] = name

        # Spring and Labour Day
        name = "Spring and Labour Day"
        self[date(year, 5, 1)] = name

        # Victory Day
        name = "Victory Day"
        self[date(year, 5, 9)] = name

        # Independence Day
        name = "Independence Day"
        self[date(year, 7, 3)] = name

        # October Revolution Day
        name = "October Revolution Day"
        self[date(year, 11, 7)] = name

        # Dec. 25 Christmas Day
        name = "Christmas Day"
        self[date(year, 12, 25)] = name


class BY(Belarus):
    pass
