#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time        : 2018/11/23 13:13
# @Author      : Spareribs
# @File        : two-sum.py
# @Software    : PyCharm
# @Description : 
"""


class Solution:
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """

        # 从后往前遍历每个数字，除了最后一个数字外
        # 如果当前的数不小于后一个数字，则加上它的值；否则减掉它的值
        luoma = {
            "I": 1,
            "V": 5,
            "X": 10,
            "L": 50,
            "C": 100,
            "D": 500,
            "M": 1000
        }

        _sum = 0
        last_num = 0
        for _id, letter in enumerate(s[::-1]):
            if _id == 0:
                _sum = luoma[letter]
                last_num = luoma[letter]
            elif luoma[letter] > last_num:
                _sum += luoma[letter]
        return _sum
